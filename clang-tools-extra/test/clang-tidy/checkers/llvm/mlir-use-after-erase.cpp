// RUN: %check_clang_tidy %s llvm-mlir-use-after-erase %t

#include <algorithm>
#include <vector>
namespace mlir {

class BitVector {};
class ValueRange {};
class Location {};

class Operation {
public:
    void destroy();
    void erase();

    void dump();
    Location getLoc() {
        return Location{};
    }
};

class OpState {
public:
  Operation *operator->() const { return state; }
  Operation *getOperation() { return state; }

private:
  Operation *state;
};
class Op : public OpState {};

class Builder {};
class OpBuilder : public Builder {};
class RewriterBase : public OpBuilder {
public:
    virtual ~RewriterBase() = default;

    virtual void eraseOp(Operation *op);
    Operation *eraseOpResults(Operation *op, const BitVector &eraseIndices);

    virtual void replaceOp(Operation *op, ValueRange newValues);
    virtual void replaceOp(Operation *op, Operation *newOp);
    template <typename OpTy, typename... Args>
    OpTy replaceOpWithNewOp(Operation *op, Args &&...args);
};
class PatternRewriter : public RewriterBase {};

} // namespace mlir

namespace test {
class MyOp : public mlir::Op {
};


} // namespace test

namespace {

void consume([[maybe_unused]] mlir::Operation *op) {
}

struct NoReturnDestructor {
    [[noreturn]] ~NoReturnDestructor();
};
#define FATAL() NoReturnDestructor()

// A type that has erase() and destroy() methods, but is not related to mlir::Operation.
class NotAnOperation {
public:
    void erase() {}
    void destroy() {}
    void dump() {}
};

// A type that has operator-> and getOperation() that return mlir::Operation*,
// but is not derived from mlir::OpState.
class NotAnOp {
public:
    mlir::Operation* operator->() const { return op; }
    mlir::Operation* getOperation() const { return op; }
private:
    mlir::Operation* op;
};

// A type that has method names like mlir::RewriterBase, but is not derived from mlir::RewriterBase.
class NotARewriter {
public:
    void eraseOp(mlir::Operation *op) {}
    void replaceOp(mlir::Operation *op, mlir::Operation *newOp) {}
};

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// General tests

void methodAccessAfterDestroyIsUse() {
    mlir::Operation* op{};
    op->destroy();
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'destroy'
    // CHECK-MESSAGES: :[[@LINE-3]]:9: note: invalidation occurred here
}

void methodAccessAfterEraseIsUse() {
    mlir::Operation* op{};
    op->erase();
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'erase'
    // CHECK-MESSAGES: :[[@LINE-3]]:9: note: invalidation occurred here
}

void derefAfterEraseIsUse() {
    mlir::Operation* op{};
    op->erase();
    *op;
    // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: 'op' used after it was invalidated by 'erase'
    // CHECK-MESSAGES: :[[@LINE-3]]:9: note: invalidation occurred here
}

void subscriptAccessAfterEraseIsUse() {
    mlir::Operation* op{};
    op->erase();
    op[0];
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'erase'
    // CHECK-MESSAGES: :[[@LINE-3]]:9: note: invalidation occurred here
}

void noAccessAfterEraseIsUse() {
    mlir::Operation* op{};
    op->erase();
    op;
    std::vector<mlir::Operation*> ops{op};
    (void)std::find(ops.begin(), ops.end(), op);
}

void methodAccessAfterRewriterEraseIsUse() {
    mlir::RewriterBase rewriter{};
    mlir::Operation* op{};
    rewriter.eraseOp(op);
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'eraseOp'
    // CHECK-MESSAGES: :[[@LINE-3]]:14: note: invalidation occurred here
}

void methodAccessAfterRewriterEraseOpResultsIsUse() {
    mlir::RewriterBase rewriter{};
    mlir::Operation* op{};
    rewriter.eraseOpResults(op, {});
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'eraseOpResults'
    // CHECK-MESSAGES: :[[@LINE-3]]:14: note: invalidation occurred here
}

void methodAccessAfterRewriterReplaceOpIsUse() {
    mlir::RewriterBase rewriter{};
    {
        mlir::Operation* op{};
        rewriter.replaceOp(op, {});
        op->dump();
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'op' used after it was invalidated by 'replaceOp'
        // CHECK-MESSAGES: :[[@LINE-3]]:18: note: invalidation occurred here
    }
    {
        mlir::Operation* op{};
        mlir::Operation* op2{};
        rewriter.replaceOp(op, op2);
        op->dump();
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'op' used after it was invalidated by 'replaceOp'
        // CHECK-MESSAGES: :[[@LINE-3]]:18: note: invalidation occurred here
    }
}

void methodAccessAfterRewriterReplaceOpWithNewOpIsUse() {
    mlir::RewriterBase rewriter{};
    mlir::Operation* op{};
    rewriter.replaceOpWithNewOp<mlir::Operation>(op);
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'replaceOpWithNewOp<mlir::Operation>'
    // CHECK-MESSAGES: :[[@LINE-3]]:14: note: invalidation occurred here
}

void methodAccessAfterPatternRewriterEraseIsUse() {
    mlir::PatternRewriter rewriter{};
    mlir::Operation* op{};
    rewriter.eraseOp(op);
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'eraseOp'
    // CHECK-MESSAGES: :[[@LINE-3]]:14: note: invalidation occurred here
}

// void useAfterDerivedArrowErase() {
//     test::MyOp op{};
//     op->erase();
//     op->dump();
//     // :[[@LINE-1]]:5: warning: operation 'op' is used after it was erased
//     // :[[@LINE-3]]:9: note: operation erased here
// }

// void useAfterDerivedGetOperationErase() {
//     test::MyOp op{};
//     op.getOperation()->erase();
//     op->dump();
//     // :[[@LINE-1]]:5: warning: operation 'op' is used after it was erased
//     // :[[@LINE-3]]:24: note: operation erased here
// }

// void useAfterRewriterEraseDerived() {
//     mlir::RewriterBase rewriter{};
//     test::MyOp op{};
//     rewriter.eraseOp(op.getOperation());
//     op->dump();
//     // :[[@LINE-1]]:5: warning: operation 'op' is used after it was erased
//     // :[[@LINE-3]]:14: note: operation erased here
// }

// void onlyFlagOneUseAfterErase() {
//     mlir::Operation* op{};
//     op->erase();
//     op->dump();  // A warning should only be emitted for one use-after-erase
//     // :[[@LINE-1]]:5: warning: operation 'op' is used after it was erased
//     // :[[@LINE-3]]:9: note: operation erased here
//     op->dump();
// }

void eraseAfterErase() {
    mlir::Operation* op{};
    op->erase();
    op->erase();  // Erase-after-erase also counts as a use
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'erase'
    // CHECK-MESSAGES: :[[@LINE-3]]:9: note: invalidation occurred here
}

void useInCall() {
    mlir::Operation* op{};
    op->erase();
    consume(op);  // Only dereferences and member accesses are flagged as uses
}

void useParameterAfterErase(mlir::Operation* op) {
    op->erase();
    op->dump();  // The erased operation may be a function parameter
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'erase'
    // CHECK-MESSAGES: :[[@LINE-3]]:9: note: invalidation occurred here
}

struct Container {
    void useAfterEraseInMemberFunction() {
        mlir::Operation* op{};
        op->erase();
        op->dump();  // The check also works in member functions
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'op' used after it was invalidated by 'erase'
        // CHECK-MESSAGES: :[[@LINE-3]]:13: note: invalidation occurred here
    }
};

void useAfterEraseInLambda() {
    [] {
        mlir::Operation* op{};
        op->erase();
        op->dump();  // The check also works in lambdas
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'op' used after it was invalidated by 'erase'
        // CHECK-MESSAGES: :[[@LINE-3]]:13: note: invalidation occurred here
    }();
}

// Using an operation before it is erased is fine
void useBeforeErase() {
    {
        mlir::Operation* op{};
        op->dump();
        op->destroy();
    }
    {
        mlir::PatternRewriter rewriter{};
        mlir::Operation* op{};
        op->dump();
        rewriter.eraseOp(op);
    }
}

// void derivedUseBeforeErase() {
//     test::MyOp op{};
//     op->dump();  // Use before erase is fine
//     op->erase();
// }

////////////////////////////////////////////////////////////////////////////////
// Tests involving control flow

void useAndEraseInLoop(int size) {
    mlir::Operation* op{};
    for (int i = 0; i < size; ++i) {
        op->dump();
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'op' used after it was invalidated by 'erase'
        // CHECK-MESSAGES: :[[@LINE+2]]:13: note: invalidation occurred here
        // CHECK-MESSAGES: :[[@LINE-3]]:9: note: the use happens in a later loop iteration than the invalidation
        op->erase();
    }
}

// void derivedUseAndEraseInLoop(int size) {
//     test::MyOp op{};
//     for (int i = 0; i < size; ++i) {
//         op->dump();
//         //: :[[@LINE-1]]:9: warning: operation 'op' is used after it was erased
//         //: :[[@LINE+2]]:13: note: operation erased here
//         //: :[[@LINE-3]]:9: note: the use happens in a later loop iteration than the erase
//         op->erase();
//     }
// }

void declaredInLoop() {
    for (int i = 0; i < 10; ++i) {
        mlir::Operation* op{};
        op->dump();
        op->erase();  // No warning if 'op' is declared inside the loop, because it is a fresh operation on every iteration
    }
}

void returnAfterErase(int i) {
    mlir::Operation* op{};
    for (int j = 0; j < 10; ++j) {
        op->dump();
        if (i > 0) {
            op->erase();  // Don't warn if we return after the erase
            return;
        }
    }
}

void differentBranches(int i) {
    mlir::Operation* op{};
    if (i > 0) {
        op->erase();
    } else {
        op->dump();  // Don't warn if the use is in a different branch from the erase
    }
}

void differentBranchesRewriter(int i) {
    mlir::PatternRewriter rewriter{};
    mlir::Operation* op{};
    if (i > 0) {
        rewriter.eraseOp(op);
    } else {
        op->dump();  // Don't warn if the use is in a different branch from the erase
    }
}

void switchFallthrough(int i) {
    mlir::Operation* op{};
    switch (i) {
    case 1:
        op->erase();
    case 2:  // A fallthrough in a switch statement causes a warning
        op->dump();
        // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'op' used after it was invalidated by 'erase'
        // CHECK-MESSAGES: :[[@LINE-4]]:13: note: invalidation occurred here
        break;
    }
}

void noReturnDestructorAfterErase(bool cond, mlir::Operation* other) {
    mlir::Operation* op{};
    op->erase();
    if (cond) {
        FATAL();  // [[noreturn]] destructor
    } else {
        op = other;
    }
    op->dump(); // only reachable through the branch that reassigns 'op', so no warning should be emitted
}

////////////////////////////////////////////////////////////////////////////////
// Tests for reinitializations

void reassignAfterErase(mlir::Operation* other) {
    mlir::Operation* op{};
    op->erase();
    op = other;  // Reassignment to a new operation makes the later use safe
    op->dump();
}

void derivedReassignAfterErase(test::MyOp other) {
    test::MyOp op{};
    op->erase();
    op = other;  // Reassignment to a new operation makes the later use safe
    op->dump();
}

void conditionalReassign(int i, mlir::Operation* other) {
    mlir::Operation* op{};
    op->erase();
    if (i > 0) {
        op = other;  // Reassignment is not guaranteed to happen before the use, so we still warn
    }
    op->dump();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'op' used after it was invalidated by 'erase'
    // CHECK-MESSAGES: :[[@LINE-6]]:9: note: invalidation occurred here
}

////////////////////////////////////////////////////////////////////////////////
// Tests related to order of evaluation within expressions

// Not reported in C++17
// void sequencingOfEraseAndUse() {
//     const auto fn = [](mlir::Operation, mlir::Location) {
//     };
//     mlir::RewriterBase rewriter{};
//     mlir::Operation* op{};
//     fn(rewriter.replaceOpWithNewOp<mlir::Operation>(op), op->getLoc());
//     // :[[@LINE-1]]:8: warning: 'op' used after it was invalidated by 'replaceOpWithNewOp<mlir::Operation>'
//     // :[[@LINE-2]]:22: note: invalidation occurred here
//     // :[[@LINE-3]]:8: note: the use and invalidation are unsequenced, i.e. there is no guarantee about the order in which they are evaluated
// }

////////////////////////////////////////////////////////////////////////////////
// Tests for types and operations that are not tracked

void eraseOnNonOperation() {
    NotAnOperation* obj{};
    obj->erase();  // erase() on an unrelated type are not flagged
    obj->dump();
}

void destroyOnNonOperation() {
    NotAnOperation* obj{};
    obj->destroy();  // destroy() on an unrelated type is not flagged
    obj->dump();
}

void eraseOpOnNonRewriter() {
    NotARewriter rewriter{};
    mlir::Operation* op{};
    rewriter.eraseOp(op);  // eraseOp() on a type not derived from RewriterBase is not flagged
    op->dump();
}

void replaceOpOnNonRewriter() {
    NotARewriter rewriter{};
    mlir::Operation* op{};
    mlir::Operation* newOp{};
    rewriter.replaceOp(op, newOp);  // replaceOp() on a type not derived from RewriterBase is not flagged
    op->dump();
}

// void eraseOnNonOpState() {
//     NotAnOp op{};
//     op->erase();  // operator-> on a type not derived from mlir::OpState is not flagged
//     op->dump();
// }

// void eraseGetOperationOnNonOpState() {
//     NotAnOp op{};
//     op.getOperation()->erase();  // getOperation() on a type not derived from mlir::OpState is not flagged
//     op->dump();
// }

mlir::Operation* globalOp{};  // Operations that are not local variables are not tracked
void eraseGlobalOperation() {
    globalOp->erase();
    globalOp->dump();
}

struct OperationHolder {
    mlir::Operation* op;  // Operations stored in member fields are not tracked
    void eraseAndUse() {
        op->erase();
        op->dump();
    }
};

void differentOperations() {
    mlir::Operation* op1{};
    mlir::Operation* op2{};
    op1->erase();
    op2->dump();  // Erasing one operation doesn't flag the use of a different operation
}

void replacementOperationNotErased() {
    mlir::RewriterBase rewriter{};
    mlir::Operation* op{};
    mlir::Operation* newOp{};
    rewriter.replaceOp(op, newOp);  // The replacement operation passed to replaceOp is not the erased one
    newOp->dump();
}

void decltypeIsNotUse() {
    mlir::Operation* op{};
    op->erase();
    using OpType = decltype(op);  // Using decltype on an erased operation is not flagged
    OpType op2{};
    op2->dump();
}

void unevaluatedIsNotUse() {
    mlir::Operation* op{};
    op->erase();
    static_assert(sizeof(op) == sizeof(void*)); // Unevaluated, so not a real use
}
