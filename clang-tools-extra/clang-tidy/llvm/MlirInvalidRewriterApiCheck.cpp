//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MlirInvalidRewriterApiCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void MlirInvalidRewriterApiCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(hasName("matchAndRewrite")).bind("matchAndRewrite"), this);
}

void MlirInvalidRewriterApiCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FuncDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("matchAndRewrite");
  if (!FuncDecl)
    return;
}

} // namespace clang::tidy::llvm_check
