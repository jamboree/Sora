import os
import sys
import lit
import lit.formats
import subprocess

config.name = 'Sora Unit Tests'
config.suffixes = ['.sora']
config.test_format = lit.formats.ShTest(execute_external=False)
config.test_source_root = os.path.dirname(__file__)

# Check if sorac is available
try:
  subprocess.check_output(["sorac", "-h"])
except OSError as e:
  lit_config.error("Can't find the Sora binary in lit parameters or PATH")
  exit(1)

# Check if FileCheck is available
try:
  subprocess.check_output(["FileCheck", "--help"])
except OSError as e:
  lit_config.error("Can't find LLVM FileCheck")
  exit(1)

for (param, value) in lit_config.params.items():
  if param == "FileCheck":
    config.substitutions.append((param, value))
