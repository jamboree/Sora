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
class BufferID;
class DiagnosticEngine;
class InFlightDiagnostic;
class SourceManager;

/// Represent a diagnostic string with its kind and argument types as template
/// parameters.
///
/// This should usually be passed by const-reference.
///
/// This is a helpful tool because it allows diagnostics to be type-safe,
/// reducing the likelihood of a formatting error due to incorrect argument
/// count/types.
template <typename... ArgTypes> struct TypedDiag {
  const DiagnosticKind kind;
  const char *const str;

  TypedDiag(DiagnosticKind kind, const char *str) : kind(kind), str(str) {
    assert(str && "Diagnostic string cannot be null!");
  }
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

/// This is the class that can be specialized to support custom types
/// in Diagnostics.
///
/// For instance, if the "Foo" class wants to be usable in diagnostics,
/// It needs to specialize this class for "Foo", and provide a definition
/// of the static format method that returns a human-readable string
/// usable in diagnostics. SORA_FWD_DECL(class Foo) will also need to be used in
/// the files that wish to use Foo as diagnostic arguments.
template <typename Ty> struct DiagnosticArgument {
  static std::string format(Ty value) {
    // This below will always evaluate to false and trip when ::format
    // doesn't exist for that type
    static_assert(!std::is_same<Ty, Ty>::value,
                  "No specialization of DiagnosticArgument for this type.");
  }

  // Add a specialization for size_t in case it differs from "unsigned int" on
  // some systems.
};

/// Provide some implementation of DiagnosticArgument for common types.
#define DIAGNOSTIC_ARGUMENT(Ty, PassTy, Body)                                  \
  template <> struct DiagnosticArgument<Ty> {                                  \
    static std::string format(Ty value) Body                                   \
  }
DIAGNOSTIC_ARGUMENT(size_t, size_t, { return std::to_string(value); });
DIAGNOSTIC_ARGUMENT(int, int, { return std::to_string(value); });
DIAGNOSTIC_ARGUMENT(char, char, { return std::string(1, value); });
DIAGNOSTIC_ARGUMENT(StringRef, StringRef, { return value.str(); });
DIAGNOSTIC_ARGUMENT(std::string, const std::string &, { return value; });
#undef DIAGNOSTIC_ARGUMENT

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
  diagnose(BufferID buffer, const TypedDiag<Args...> &diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagnose(getLocForDiag(buffer), diag, std::forward(args)...);
  }

  /// Emits a \p diag at \p loc with \p args
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, const TypedDiag<Args...> &diag,
           typename detail::PassArgument<Args>::type... args);

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

  /// An Argument-providing function.
  /// Those are created by binding an argument's DiagnosticArgument::format
  /// function with the argument.
  using ArgProviderFn = std::function<std::string()>;

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

  /// Formats \p diagStr using \p providers and returns the result.
  std::string formatDiagnosticString(const char *diagStr,
                                     ArrayRef<ArgProviderFn> providers);

  /// The Diagnostic Consumer
  std::unique_ptr<DiagnosticConsumer> consumer = nullptr;

  /// Contains the data of a diagnostic.
  struct DiagnosticData {
    DiagnosticData(const DiagnosticData &) = delete;
    DiagnosticData &operator=(const DiagnosticData &) = delete;

    /// Constructor for diagnostics with arguments
    template <typename... Args>
    DiagnosticData(const TypedDiag<Args...> &diag, SourceLoc loc,
                   typename detail::PassArgument<Args>::type... args)
        : str(diag.str), kind(diag.kind), loc(loc) {
      argProviders = {std::bind(DiagnosticArgument<Args>::format, args)...};
    }

    /// Constructor for diagnostics with no arguments
    template <typename... Args>
    DiagnosticData(const TypedDiag<> &diag, SourceLoc loc)
        : str(diag.str), kind(diag.kind), loc(loc) {}

    SmallVector<ArgProviderFn, 4> argProviders;
    const char *str;
    const DiagnosticKind kind;
    SmallVector<FixIt, 2> fixits;
    const SourceLoc loc;
    SmallVector<CharSourceRange, 2> ranges;
  };

  /// The currently active diagnostic, or "None" if there is no
  /// active diagnostic.
  Optional<DiagnosticData> activeDiagnostic;
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
  DiagnosticEngine::DiagnosticData &getDiagnosticData();

  const DiagnosticEngine::DiagnosticData &getDiagnosticData() const {
    return const_cast<InFlightDiagnostic *>(this)->getDiagnosticData();
  }

  /// Converts a SourceRange to a CharSourceRange
  CharSourceRange toCharSourceRange(SourceRange range) const;

  /// \returns true if this diagnostic is active and has a valid loc
  bool canAddInfo() const;

public:
  InFlightDiagnostic() = default;

  InFlightDiagnostic(const InFlightDiagnostic &) = delete;
  InFlightDiagnostic &operator=(const InFlightDiagnostic &) = delete;

  InFlightDiagnostic(InFlightDiagnostic &&other) { *this = std::move(other); }

  InFlightDiagnostic &operator=(InFlightDiagnostic &&other) {
    diagEngine = other.diagEngine;
    other.diagEngine = nullptr;
    return *this;
  }

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

  /// Adds a insertion fix-it (insert \p text after the token at \p loc)
  InFlightDiagnostic &fixitInsertAfter(SourceLoc loc, StringRef text);

  /// Adds a replacement fix-it (replace the character range \p range by
  /// \p text)
  InFlightDiagnostic &fixitReplace(CharSourceRange range, StringRef text);

  /// Adds a removal fix-it (remove the tokens in \p range)
  InFlightDiagnostic &fixitRemove(SourceRange range);
};

template <typename... Args>
InFlightDiagnostic
DiagnosticEngine::diagnose(SourceLoc loc, const TypedDiag<Args...> &diag,
                           typename detail::PassArgument<Args>::type... args) {
  assert(!activeDiagnostic.hasValue() && "A diagnostic is already in-flight!");
  activeDiagnostic.emplace(diag, loc, std::move(args)...);
  return InFlightDiagnostic(this);
}
} // namespace sora
