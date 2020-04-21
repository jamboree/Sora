#!/usr/bin/env python3

"""
    This is a script to update tests relying on FileCheck.
    This is a fairly naive script - it just appends the lines in stdin to 
    the bottom of the file with the check prefix.

    Use it like this: lit <tests_you_want_to_update> -DFileCheck=<path to this script>
    or -DFileCheck="python <path to this script>" on Windows.

    Do not update every test with this script! 
    Only update those that you need to update, and 
    never use this script to update tests that were not auto-generated!
"""

import sys
import os
import argparse


def get_comment(x: str):
    return "// " + x


FILE_HEADER = get_comment('NOTE: CHECK lines have been generated by utils/' + os.path.basename(__file__))


def fetch_check_lines_from_stdin(check_prefix: str) -> [str]:
    check_lines: [str] = []
    use_check_next = False
    for line in sys.stdin:
        line = line.rstrip()
        if line == "":
            continue
        if not use_check_next:
            # The added spaces are to keep the OCD in check :)
            line = check_prefix + ':      ' + line
        else:
            line = check_prefix + '-NEXT: ' + line
            use_check_next = False
        check_lines.append(line)
    return check_lines


def read_lines(file: str) -> [str]:
    lines: [str]
    with open(file) as file:
        lines = [line.rstrip() for line in file]
    return lines


def rewrite_file(file: str, lines: [str]):
    with open(file, 'w') as file:
        for line in lines:
            file.write(line + "\n")


def insert_file_header(lines: [str]):
    if not lines[0].startswith(FILE_HEADER):
        lines.insert(0, FILE_HEADER)


def remove_old_check_lines(lines: [str], check_prefix: str) -> [str]:
    return [l for l in lines if not l.strip().startswith(get_comment(check_prefix))]


def append_check_lines(lines: [str], check_lines: [str]):
    if lines[-1] != "":
        lines.append("")
    for check_line in check_lines:
        lines.append(get_comment(check_line))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('testfile', metavar="<test file>", type=str)
    args = parser.parse_args()
    testfile = args.testfile

    # TODO: Support --check-prefix
    check_prefix = 'CHECK'

    check_lines: [str] = fetch_check_lines_from_stdin(check_prefix)

    file_lines = read_lines(testfile)

    insert_file_header(file_lines)
    file_lines = remove_old_check_lines(file_lines, check_prefix)
    append_check_lines(file_lines, check_lines)

    rewrite_file(testfile, file_lines)


if __name__ == '__main__':
    main()
    exit(0)
