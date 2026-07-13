// RUN: %check_clang_tidy %s llvm-mlir-invalid-rewriter-api %t

namespace mlir {

template <typename Fn>
class function_ref;
template <typename PtrType>
class SmallPtrSetImpl;
template <typename T>
class SmallVectorImpl;
class BitVector;

class Block {
public:
    using iterator = char*;
};
class LogicalResult {};
class Operation {};
class OpOperand {};
class Value {};
class ValueRange {};
class Pattern {};
class Region {
public:
    using iterator = char*;
};

class Builder {};
class OpBuilder : public Builder {};
class RewriterBase : public OpBuilder {
public:
    virtual ~RewriterBase() = default;

    virtual void replaceOp(Operation *op, ValueRange newValues);
    virtual void replaceOp(Operation *op, Operation *newOp);
    template <typename OpTy, typename... Args>
    OpTy replaceOpWithNewOp(Operation *op, Args &&...args);

    virtual void eraseOp(Operation *op);
    virtual void eraseBlock(Block *block);
    Operation *eraseOpResults(Operation *op, const BitVector &eraseIndices);

    void moveOpBefore(Operation *op, Operation *existingOp);
    void moveOpBefore(Operation *op, Block *block, Block::iterator iterator);
    void moveOpAfter(Operation *op, Operation *existingOp);
    void moveOpAfter(Operation *op, Block *block, Block::iterator iterator);
    void moveBlockBefore(Block *block, Block *anotherBlock);
    void moveBlockBefore(Block *block, Region *region, Region::iterator iterator);

    virtual void replaceAllUsesWith(Value from, Value to);
    void replaceAllUsesWith(Block *from, Block *to);
    void replaceAllUsesWith(ValueRange from, ValueRange to);
    void replaceAllOpUsesWith(Operation *from, ValueRange to);
    void replaceAllOpUsesWith(Operation *from, Operation *to);
    void replaceUsesWithIf(ValueRange from, ValueRange to,
                           function_ref<bool(OpOperand &)> functor,
                           bool *allUsesReplaced = nullptr);
    void replaceOpUsesWithIf(Operation *from, ValueRange to,
                             function_ref<bool(OpOperand &)> functor,
                             bool *allUsesReplaced = nullptr);
    void replaceOpUsesWithinBlock(Operation *op, ValueRange newValues,
                                  Block *block, bool *allUsesReplaced = nullptr);
    void replaceAllUsesExcept(Value from, Value to, Operation *exceptedUser);
    void replaceAllUsesExcept(Value from, Value to,
                              const SmallPtrSetImpl<Operation *> &preservedUsers);

    LogicalResult tryFold(Operation *op, SmallVectorImpl<Value> &results,
                          SmallVectorImpl<Operation *> *materializedConstants = nullptr);
};
class PatternRewriter : public RewriterBase {};

class RewritePattern : public Pattern {
public:
    virtual ~RewritePattern() = default;
    virtual LogicalResult matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const = 0;
};

namespace detail {
template <typename SourceOp>
struct OpOrInterfaceRewritePatternBase : public RewritePattern {
    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const final {
      return LogicalResult();
    }
};
} // namespace detail

template <typename SourceOp>
struct OpRewritePattern
    : public mlir::detail::OpOrInterfaceRewritePatternBase<SourceOp> {
};

} // namespace mlir

template<typename OpTy>
struct InvalidIRModifications final : mlir::OpRewritePattern<OpTy> {
  mlir::LogicalResult matchAndRewrite(OpTy op,
                                      mlir::PatternRewriter &rewriter) const override {
    op->clone();
    op->cloneWithoutRegions();
    op->destroy();
    op->dropAllDefinedValueUses();
    op->dropAllUses();
    op->erase();
    op->eraseOperand();
    op->eraseOperands();
    op->fold();
    op->moveAfter();
    op->moveBefore();
    op->remove();
    op->removeAttr();
    op->removeDiscardableAttr();
    op->replaceAllUsesWith();
    op->replaceUsesOfWith();
    op->replaceUsesWithIf();
    return mlir::LogicalResult();
  }
};

template<typename OpTy>
struct ValidIRModifications final : mlir::OpRewritePattern<OpTy> {
  mlir::LogicalResult matchAndRewrite(OpTy op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Block* block = nullptr;

    rewriter.eraseOp(op);
    rewriter.eraseBlock(block);
    rewriter.eraseOpResults(op, {});
    rewriter.tryFold(op, {});
    rewriter.moveBlockBefore(block, block);
    rewriter.moveOpBefore(op, op);
    rewriter.moveOpAfter(op, op);
    rewriter.replaceAllUsesExcept(mlir::Value(), mlir::Value(), op);
    rewriter.replaceAllUsesWith(mlir::Value(), mlir::Value());
    rewriter.replaceOpUsesWithIf(op, mlir::ValueRange(), nullptr);
    rewriter.replaceOpUsesWithinBlock(op, mlir::ValueRange(), block);
    // rewriter.replaceUsesWithIf(mlir::ValueRange(), mlir::ValueRange(), nullptr);
    rewriter.replaceAllUsesExcept(mlir::Value(), mlir::Value(), op);
    rewriter.replaceOpWithNewOp<OpTy>(op);
    rewriter.replaceOp(op, op);
    return mlir::LogicalResult();
  }
};
