//===--- DiagnosticEngine.hpp - Diagnostic wrangling ------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Contains the DiagnosticEngine and related classes.
//
// TODO: Describe the Diagnostic System more in-depth.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticConsumer.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <stdint.h>

namespace sora {
class DiagnosticEngine;
class SourceManager;
class BufferID;

/// The unique identifier of a diagnostic
enum class DiagID : uint32_t;

/// Wrapper around a DiagID with additional template parameter describing the
/// type of the arguments required by the diagnostic.
///
/// This is a helpful tool because it allows diagnostics to be mostly type-safe,
/// ensuring that no formatting error can occur due to mismatched types.
///
/// FIXME: Maybe this can have a better name?
template <typename... ArgTypes> struct TypedDiag {
  /// The unique id of the diagnostic
  DiagID id;
};

namespace detail {
/// Helper structure that describes how to pass a diagnostic argument of a
/// given type to a function. This can be specialized to pass larger argument
/// types by reference.
///
/// This is also used for some template trickery, because it allows the
/// compiler to make implicit conversions in scenarios like this:
/// \code
///   diagEng.diagnose(loc, diag::foo, "str").
/// \encode
/// Here, "foo" expects a StringRef, so the const char* is automatically
/// converted into a StringRef. This isn't possible without the PassArgument
/// struct for whatever reason. The compiler (MSVC) will complain
/// that the argument is ambiguous and that it could be a const char* or
/// StringRef. Maybe it's just an MSVC thing?
template <typename Ty> struct PassArgument { using type = Ty; };
} // namespace detail

/// This is the class that should be specialized to support custom types
/// in Diagnostics.
///
/// For instance, if the "Foo" class wants to be usable in diagnostics,
/// It needs to specialize this class for "Foo", and provide a definition
/// of the static format method that returns a human-readable string
/// usable in diagnostics.
template <typename Ty> struct DiagnosticArgumentFormatter {
  static std::string format(Ty value) {
    // This below will always evaluate to false and trip when ::format 
    // doesn't exist for that type
    static_assert(!std::is_same<Ty, Ty>::value,
        "No specialization of DiagnosticArgumentFormatter for this type.");
  }
};

/// Provide some implementation for basic types.
/// FIXME: Can this be implemented better? For now it just works, but I feel
/// like there's a better, template-y way to do this.
#define SIMPLE_DAF_IMPL(Ty, PassTy, Body)                                      \
  template <> struct DiagnosticArgumentFormatter<Ty> {                         \
    static std::string format(Ty value) Body                                   \
  }
SIMPLE_DAF_IMPL(unsigned, unsigned, { return std::to_string(value); });
SIMPLE_DAF_IMPL(int, int, { return std::to_string(value); });
SIMPLE_DAF_IMPL(char, char, { return std::string(1, value); });
SIMPLE_DAF_IMPL(StringRef, StringRef, { return value.str(); });
SIMPLE_DAF_IMPL(std::string, const std::string &, { return value; });
#undef SIMPLE_DAF_IMPL

/// The type of a Diagnostic Argument "Provider".
/// This is created by binding the DiagnosticArgumentFormatter::format
/// static method to the argument value.
///
/// This is pretty helpful because it makes argument formatting very lazy. If a
/// diagnostic is aborted, we won't even format its argument and the diagnostic
/// string!
///
/// e.g. if the type is "Foo", then this is created by doing
/// \verbatim
///   std::bind(DiagnosticArgumentFormatter<Foo>::format, arg)
/// \endverbatim
using DiagnosticArgumentProvider = std::function<std::string()>;

/// Represents a raw, unformatted Diagnostic. This is used to store
/// the data of in-flight diagnostics.
class RawDiagnostic {
  SmallVector<DiagnosticArgumentProvider, 4> argProviders;
  const DiagID id;
  SmallVector<FixIt, 2> fixits;
  const SourceLoc loc;
  SmallVector<CharSourceRange, 2> ranges;

  friend DiagnosticEngine;

public:
  RawDiagnostic(const RawDiagnostic &) = delete;
  RawDiagnostic &operator=(const RawDiagnostic &) = delete;

  /// Constructor for diagnostics with arguments
  template <typename... Args>
  RawDiagnostic(TypedDiag<Args...> diag, SourceLoc loc,
                typename detail::PassArgument<Args>::type... args)
      : id(diag.id), loc(loc) {
    argProviders = {
        std::bind(DiagnosticArgumentFormatter<Args>::format, args)...};
  }

  /// Constructor for diagnostics with no arguments
  template <typename... Args>
  RawDiagnostic(TypedDiag<> diag, SourceLoc loc) : id(diag.id), loc(loc) {}

  /// Adds a FixIt object to this Diagnostic.
  void addFixit(const FixIt &fixit) {
    assert(loc && "Cannot add a FixIt to a diagnostic if it does not "
                  "have a SourceLoc!");
    fixits.push_back(fixit);
  }
  /// Adds a CharSourceRange object to this Diagnostic.
  void addRange(const CharSourceRange &range) {
    assert(loc && "Cannot add a SourceRange to a diagnostic if it does not "
                  "have a SourceLoc!");
    ranges.push_back(range);
  }

  /// \returns The argument providers for this Diagnostic.
  ArrayRef<DiagnosticArgumentProvider> getArgProviders() const {
    return argProviders;
  }
  /// \returns The location of this Diagnostic.
  SourceLoc getLoc() const { return loc; }
  /// \returns The FixIts attached to this Diagnostic.
  ArrayRef<FixIt> getFixits() const { return fixits; }
  /// \returns The ID of this Diagnostic.
  DiagID getDiagID() const { return id; }
  /// \returns The additional ranges of this Diagnostic.
  ArrayRef<CharSourceRange> getRanges() const { return ranges; }
};

/// Builder for in-flight diagnostics (attach Fix-Its and highlight
/// additional ranges of text)
///
/// When this object is destroyed, it emits the Diagnostic.
/// The Diagnostic can be aborted by calling "abort()".
///
/// NOTE: If the diagnostic doesn't have a valid SourceLoc, you can't change it.
class InFlightDiagnostic {
  /// The DiagnosticEngine instance
  DiagnosticEngine *diagEngine = nullptr;

  friend DiagnosticEngine;

  /// Constructor for the DiagnosticEngine
  InFlightDiagnostic(DiagnosticEngine *diagEngine) : diagEngine(diagEngine) {}

  /// \returns the raw diagnostic we're building
  RawDiagnostic &getRawDiagnostic();

  const RawDiagnostic &getRawDiagnostic() const {
    return const_cast<InFlightDiagnostic *>(this)->getRawDiagnostic();
  }

  /// Converts a SourceRange to a CharSourceRange
  CharSourceRange toCharSourceRange(SourceRange range) const;

  /// \returns true if this diagnostic is active and has a valid loc
  bool canAddInfo() const;

public:
  InFlightDiagnostic() = default;

  /// Emits the Diagnostic
  ~InFlightDiagnostic();

  /// Aborts this diagnostic (it will not be emitted)
  void abort();

  /// \returns true if this diagnostic is still active
  bool isActive() const { return diagEngine; }

  /// Highlights the range of characters covered by \p range
  InFlightDiagnostic &highlightChars(CharSourceRange range);

  /// Highlights the range of tokens \p range
  InFlightDiagnostic &highlight(SourceRange range);

  /// Adds a insertion fix-it (insert \p text at \p loc)
  InFlightDiagnostic &fixitInsert(SourceLoc loc, StringRef text);

  /// Adds a replacement fix-it (replace the character range \p range by
  /// \p text)
  InFlightDiagnostic &fixitReplace(CharSourceRange range, StringRef text);

  /// Adds a removal fix-it (remove the tokens in \p range)
  InFlightDiagnostic &fixitRemove(SourceRange range);
};

/// The DiagnosticEngine, the heart of the Diagnostic System.
/// This handles most things related to diagnostics: creation, feeding
/// them to consumers and customization (e.g. muting all diagnostic)
class DiagnosticEngine {
  //===--- Bitfield: 5 bits left ---===//
  bool errorOccured : 1;
  bool warningsAreErrors : 1;
  bool ignoreAll : 1;
  //===-----------------------------===//

  /// Initializes the bitfields above
  void initBitfields();

  /// \returns the SourceLoc that should be used to emit a diagnostic about
  /// \p buffer. For now, this is always the beginning of the file.
  SourceLoc getLocForDiag(BufferID buffer) const;

public:
  explicit DiagnosticEngine(const SourceManager &srcMgr) : srcMgr(srcMgr) {
    initBitfields();
  }

  // The DiagnosticEngine is non-copyable.
  DiagnosticEngine(const DiagnosticEngine &) = delete;
  DiagnosticEngine &operator=(const DiagnosticEngine &) = delete;

  /// Emits a \p diag at \p loc with \p args
  template <typename... Args>
  InFlightDiagnostic
  diagnose(BufferID buffer, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagnose(getLocForDiag(buffer), diag, std::forward(args)...);
  }

  /// Emits a \p diag at \p loc with \p args
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    assert(!activeDiagnostic.hasValue() &&
           "A diagnostic is already in-flight!");
    activeDiagnostic.emplace(diag, loc, std::move(args)...);
    return InFlightDiagnostic(this);
  }

  /// \returns a observing pointer to the current diagnostic consumer
  DiagnosticConsumer *getConsumer() { return consumer.get(); }

  /// Creates a new DiagnosticConsumer to replace the current one.
  template <typename Consumer, typename... Args,
            typename = typename std::enable_if_t<
                std::is_base_of<DiagnosticConsumer, Consumer>::value, Consumer>>
  void createConsumer(Args &&... args) {
    setConsumer(std::make_unique<Consumer>(std::forward<Args>(args)...));
  }

  /// Replaces the current diagnostic consumer with \p newConsumer
  void setConsumer(std::unique_ptr<DiagnosticConsumer> newConsumer) {
    consumer = std::move(newConsumer);
  }

  /// Steals the DiagnosticConsumer from this DiagnosticEngine
  std::unique_ptr<DiagnosticConsumer> takeConsumer() {
    return std::move(consumer);
  }

  /// \returns true if at least one error was emitted
  bool hadAnyError() const { return errorOccured; }

  /// Gets the "ignore all" attribute.
  /// \returns true if all diagnostics are ignored
  bool getIgnoreAll() const { return ignoreAll; }
  /// Sets the "ignore all" attribute.
  /// \param true if all diagnostics should be ignored, false otherwise.
  void setIgnoreAll(bool value = true) { ignoreAll = value; }

  /// Gets the "warnings are errors" attribute.
  /// \returns true if warnings are treated as errors, false otherwise.
  bool getWarningsAreErrors() const { return warningsAreErrors; }
  /// Sets the "warnings are errors" attribute.
  /// \param true if warnings should be treated as errors, false otherwise.
  void setWarningsAreErrors(bool value = true) { warningsAreErrors = true; }

  /// The SourceManager instance used by this DiagnosticEngine.
  /// This will also be passed to consumers.
  const SourceManager &srcMgr;

private:
  friend InFlightDiagnostic;

  /// \returns the kind of diagnostic of \p id depending on the current
  /// state of this DiagnosticEngine. Returns "None" if the diagnostic
  /// should not be emitted.
  Optional<DiagnosticKind> getDiagnosticKind(DiagID id);

  /// Method that should be called when a diagnostic of kind \p kind
  /// is about to be emitted.
  void actOnDiagnosticEmission(DiagnosticKind kind);

  /// Emits the activeDiagnostic.
  /// For use by InFlightDiagnostic only.
  void emit();

  /// Aborts the activeDiagnostic.
  /// For use by InFlightDiagnostic only.
  void abort();

  /// \returns true if we have an active diagnostic.
  bool hasActiveDiagnostic() const { return activeDiagnostic.hasValue(); }

  /// The Diagnostic Consumer
  std::unique_ptr<DiagnosticConsumer> consumer = nullptr;

  /// The currently active diagnostic, or "None" if there is no
  /// active diagnostic.
  Optional<RawDiagnostic> activeDiagnostic;
};

} // namespace sora
