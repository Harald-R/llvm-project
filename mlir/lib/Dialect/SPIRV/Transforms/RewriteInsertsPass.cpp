//===- RewriteInsertsPass.cpp - MLIR conversion pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to rewrite sequential chains of
// `spirv::CompositeInsert` operations into `spirv::CompositeConstruct`
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/IR/Builders.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/IR/SPIRVAvailability.h.inc"

#if 1

#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h.inc"

#else

namespace mlir {
namespace spirv {
class ModuleOp;
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
class CompositeInsertOp;
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
class CompositeConstructOp;
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {

//===----------------------------------------------------------------------===//
// ::mlir::spirv::CompositeInsertOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CompositeInsertOpGenericAdaptorBase {
public:
  struct Properties {
    using indicesTy = ::mlir::ArrayAttr;
    indicesTy indices;

    auto getIndices() {
      auto &propStorage = this->indices;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setIndices(const ::mlir::ArrayAttr &propValue) {
      this->indices = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.indices == this->indices &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CompositeInsertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("spirv.CompositeInsert", odsAttrs.getContext());
  }

  CompositeInsertOpGenericAdaptorBase(CompositeInsertOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getIndicesAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().indices);
    return attr;
  }

  ::mlir::ArrayAttr getIndices();
};
} // namespace detail
template <typename RangeT>
class CompositeInsertOpGenericAdaptor : public detail::CompositeInsertOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CompositeInsertOpGenericAdaptorBase;
public:
  CompositeInsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CompositeInsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CompositeInsertOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CompositeInsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CompositeInsertOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = CompositeInsertOp, typename = std::enable_if_t<std::is_same_v<LateInst, CompositeInsertOp>>>
  CompositeInsertOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getObject() {
    return (*getODSOperands(0).begin());
  }

  ValueT getComposite() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CompositeInsertOpAdaptor : public CompositeInsertOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CompositeInsertOpGenericAdaptor::CompositeInsertOpGenericAdaptor;
  CompositeInsertOpAdaptor(CompositeInsertOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CompositeInsertOp : public ::mlir::Op<CompositeInsertOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::spirv::UsableInSpecConstantOp, ::mlir::spirv::QueryMinVersionInterface::Trait, ::mlir::spirv::QueryMaxVersionInterface::Trait, ::mlir::spirv::QueryExtensionInterface::Trait, ::mlir::spirv::QueryCapabilityInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CompositeInsertOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CompositeInsertOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("indices")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIndicesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIndicesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("spirv.CompositeInsert");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getObject() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getComposite() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getObjectMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCompositeMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::ArrayAttr getIndicesAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().indices);
  }

  ::mlir::ArrayAttr getIndices();
  void setIndicesAttr(::mlir::ArrayAttr attr) {
    getProperties().indices = attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value object, Value composite, ArrayRef<int32_t> indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value object, ::mlir::Value composite, ::mlir::ArrayAttr indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value object, ::mlir::Value composite, ::mlir::ArrayAttr indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  ::std::optional<::mlir::spirv::Version> getMinVersion();
  ::std::optional<::mlir::spirv::Version> getMaxVersion();
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> getExtensions();
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> getCapabilities();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace spirv
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::CompositeInsertOp)




namespace mlir {
namespace spirv {

//===----------------------------------------------------------------------===//
// ::mlir::spirv::ModuleOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ModuleOpGenericAdaptorBase {
public:
  struct Properties {
    using addressing_modelTy = ::mlir::spirv::AddressingModelAttr;
    addressing_modelTy addressing_model;

    auto getAddressingModel() {
      auto &propStorage = this->addressing_model;
      return ::llvm::cast<::mlir::spirv::AddressingModelAttr>(propStorage);
    }
    void setAddressingModel(const ::mlir::spirv::AddressingModelAttr &propValue) {
      this->addressing_model = propValue;
    }
    using memory_modelTy = ::mlir::spirv::MemoryModelAttr;
    memory_modelTy memory_model;

    auto getMemoryModel() {
      auto &propStorage = this->memory_model;
      return ::llvm::cast<::mlir::spirv::MemoryModelAttr>(propStorage);
    }
    void setMemoryModel(const ::mlir::spirv::MemoryModelAttr &propValue) {
      this->memory_model = propValue;
    }
    using sym_nameTy = ::mlir::StringAttr;
    sym_nameTy sym_name;

    auto getSymName() {
      auto &propStorage = this->sym_name;
      return ::llvm::dyn_cast_or_null<::mlir::StringAttr>(propStorage);
    }
    void setSymName(const ::mlir::StringAttr &propValue) {
      this->sym_name = propValue;
    }
    using vce_tripleTy = ::mlir::spirv::VerCapExtAttr;
    vce_tripleTy vce_triple;

    auto getVceTriple() {
      auto &propStorage = this->vce_triple;
      return ::llvm::dyn_cast_or_null<::mlir::spirv::VerCapExtAttr>(propStorage);
    }
    void setVceTriple(const ::mlir::spirv::VerCapExtAttr &propValue) {
      this->vce_triple = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.addressing_model == this->addressing_model &&
        rhs.memory_model == this->memory_model &&
        rhs.sym_name == this->sym_name &&
        rhs.vce_triple == this->vce_triple &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ModuleOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("spirv.module", odsAttrs.getContext());
  }

  ModuleOpGenericAdaptorBase(ModuleOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::spirv::AddressingModelAttr getAddressingModelAttr() {
    auto attr = ::llvm::cast<::mlir::spirv::AddressingModelAttr>(getProperties().addressing_model);
    return attr;
  }

  ::mlir::spirv::AddressingModel getAddressingModel();
  ::mlir::spirv::MemoryModelAttr getMemoryModelAttr() {
    auto attr = ::llvm::cast<::mlir::spirv::MemoryModelAttr>(getProperties().memory_model);
    return attr;
  }

  ::mlir::spirv::MemoryModel getMemoryModel();
  ::mlir::spirv::VerCapExtAttr getVceTripleAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::spirv::VerCapExtAttr>(getProperties().vce_triple);
    return attr;
  }

  ::std::optional<::mlir::spirv::VerCapExtAttr> getVceTriple();
  ::mlir::StringAttr getSymNameAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().sym_name);
    return attr;
  }

  ::std::optional< ::llvm::StringRef > getSymName();
  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};
} // namespace detail
template <typename RangeT>
class ModuleOpGenericAdaptor : public detail::ModuleOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ModuleOpGenericAdaptorBase;
public:
  ModuleOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ModuleOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ModuleOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ModuleOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ModuleOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = ModuleOp, typename = std::enable_if_t<std::is_same_v<LateInst, ModuleOp>>>
  ModuleOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ModuleOpAdaptor : public ModuleOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ModuleOpGenericAdaptor::ModuleOpGenericAdaptor;
  ModuleOpAdaptor(ModuleOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ModuleOp : public ::mlir::Op<ModuleOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::NoTerminator, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::OpTrait::SymbolTable, ::mlir::SymbolOpInterface::Trait, ::mlir::spirv::QueryMinVersionInterface::Trait, ::mlir::spirv::QueryMaxVersionInterface::Trait, ::mlir::spirv::QueryExtensionInterface::Trait, ::mlir::spirv::QueryCapabilityInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ModuleOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ModuleOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("addressing_model"), ::llvm::StringRef("memory_model"), ::llvm::StringRef("sym_name"), ::llvm::StringRef("vce_triple")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAddressingModelAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAddressingModelAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemoryModelAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemoryModelAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getVceTripleAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getVceTripleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("spirv.module");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::spirv::AddressingModelAttr getAddressingModelAttr() {
    return ::llvm::cast<::mlir::spirv::AddressingModelAttr>(getProperties().addressing_model);
  }

  ::mlir::spirv::AddressingModel getAddressingModel();
  ::mlir::spirv::MemoryModelAttr getMemoryModelAttr() {
    return ::llvm::cast<::mlir::spirv::MemoryModelAttr>(getProperties().memory_model);
  }

  ::mlir::spirv::MemoryModel getMemoryModel();
  ::mlir::spirv::VerCapExtAttr getVceTripleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::spirv::VerCapExtAttr>(getProperties().vce_triple);
  }

  ::std::optional<::mlir::spirv::VerCapExtAttr> getVceTriple();
  ::mlir::StringAttr getSymNameAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().sym_name);
  }

  ::std::optional< ::llvm::StringRef > getSymName();
  void setAddressingModelAttr(::mlir::spirv::AddressingModelAttr attr) {
    getProperties().addressing_model = attr;
  }

  void setAddressingModel(::mlir::spirv::AddressingModel attrValue);
  void setMemoryModelAttr(::mlir::spirv::MemoryModelAttr attr) {
    getProperties().memory_model = attr;
  }

  void setMemoryModel(::mlir::spirv::MemoryModel attrValue);
  void setVceTripleAttr(::mlir::spirv::VerCapExtAttr attr) {
    getProperties().vce_triple = attr;
  }

  void setSymNameAttr(::mlir::StringAttr attr) {
    getProperties().sym_name = attr;
  }

  void setSymName(::std::optional<::llvm::StringRef> attrValue);
  ::mlir::Attribute removeVceTripleAttr() {
      auto &attr = getProperties().vce_triple;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeSymNameAttr() {
      auto &attr = getProperties().sym_name;
      attr = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, std::optional<StringRef> name = std::nullopt);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, spirv::AddressingModel addressing_model, spirv::MemoryModel memory_model, std::optional<spirv::VerCapExtAttr> vce_triple = std::nullopt, std::optional<StringRef> name = std::nullopt);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verifyRegions();
  ::std::optional<::mlir::spirv::Version> getMinVersion();
  ::std::optional<::mlir::spirv::Version> getMaxVersion();
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> getExtensions();
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> getCapabilities();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  bool isOptionalSymbol() { return true; }

  std::optional<StringRef> getName() { return getSymName(); }

  static StringRef getVCETripleAttrName() { return "vce_triple"; }
};
} // namespace spirv
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::ModuleOp)


namespace mlir {
namespace spirv {

//===----------------------------------------------------------------------===//
// ::mlir::spirv::CompositeConstructOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CompositeConstructOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  CompositeConstructOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("spirv.CompositeConstruct", odsAttrs.getContext());
  }

  CompositeConstructOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class CompositeConstructOpGenericAdaptor : public detail::CompositeConstructOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CompositeConstructOpGenericAdaptorBase;
public:
  CompositeConstructOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CompositeConstructOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CompositeConstructOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = CompositeConstructOp, typename = std::enable_if_t<std::is_same_v<LateInst, CompositeConstructOp>>>
  CompositeConstructOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getConstituents() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CompositeConstructOpAdaptor : public CompositeConstructOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CompositeConstructOpGenericAdaptor::CompositeConstructOpGenericAdaptor;
  CompositeConstructOpAdaptor(CompositeConstructOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CompositeConstructOp : public ::mlir::Op<CompositeConstructOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::spirv::QueryMinVersionInterface::Trait, ::mlir::spirv::QueryMaxVersionInterface::Trait, ::mlir::spirv::QueryExtensionInterface::Trait, ::mlir::spirv::QueryCapabilityInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CompositeConstructOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CompositeConstructOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("spirv.CompositeConstruct");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getConstituents() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getConstituentsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange constituents);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  ::std::optional<::mlir::spirv::Version> getMinVersion();
  ::std::optional<::mlir::spirv::Version> getMaxVersion();
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> getExtensions();
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> getCapabilities();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace spirv
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::CompositeConstructOp)

#endif


namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVREWRITEINSERTSPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

namespace {

/// Replaces sequential chains of `spirv::CompositeInsertOp` operation into
/// `spirv::CompositeConstructOp` operation if possible.
class RewriteInsertsPass
    : public spirv::impl::SPIRVRewriteInsertsPassBase<RewriteInsertsPass> {
public:
  void runOnOperation() override;

private:
  /// Collects a sequential insertion chain by the given
  /// `spirv::CompositeInsertOp` operation, if the given operation is the last
  /// in the chain.
  LogicalResult
  collectInsertionChain(spirv::CompositeInsertOp op,
                        SmallVectorImpl<spirv::CompositeInsertOp> &insertions);
};

} // namespace

void RewriteInsertsPass::runOnOperation() {
  SmallVector<SmallVector<spirv::CompositeInsertOp, 4>, 4> workList;
  getOperation().walk([this, &workList](spirv::CompositeInsertOp op) {
    SmallVector<spirv::CompositeInsertOp, 4> insertions;
    if (succeeded(collectInsertionChain(op, insertions)))
      workList.push_back(insertions);
  });

  for (const auto &insertions : workList) {
    auto lastCompositeInsertOp = insertions.back();
    auto compositeType = lastCompositeInsertOp.getType();
    auto location = lastCompositeInsertOp.getLoc();

    SmallVector<Value, 4> operands;
    // Collect inserted objects.
    for (auto insertionOp : insertions)
      operands.push_back(insertionOp.getObject());

    OpBuilder builder(lastCompositeInsertOp);
    auto compositeConstructOp = builder.create<spirv::CompositeConstructOp>(
        location, compositeType, operands);

    lastCompositeInsertOp.replaceAllUsesWith(
        compositeConstructOp->getResult(0));

    // Erase ops.
    for (auto insertOp : llvm::reverse(insertions)) {
      auto *op = insertOp.getOperation();
      if (op->use_empty())
        insertOp.erase();
    }
  }
}

LogicalResult RewriteInsertsPass::collectInsertionChain(
    spirv::CompositeInsertOp op,
    SmallVectorImpl<spirv::CompositeInsertOp> &insertions) {
  auto indicesArrayAttr = cast<ArrayAttr>(op.getIndices());
  // TODO: handle nested composite object.
  if (indicesArrayAttr.size() == 1) {
    auto numElements = cast<spirv::CompositeType>(op.getComposite().getType())
                           .getNumElements();

    auto index = cast<IntegerAttr>(indicesArrayAttr[0]).getInt();
    // Need a last index to collect a sequential chain.
    if (index + 1 != numElements)
      return failure();

    insertions.resize(numElements);
    while (true) {
      insertions[index] = op;

      if (index == 0)
        return success();

      op = op.getComposite().getDefiningOp<spirv::CompositeInsertOp>();
      if (!op)
        return failure();

      --index;
      indicesArrayAttr = cast<ArrayAttr>(op.getIndices());
      if ((indicesArrayAttr.size() != 1) ||
          (cast<IntegerAttr>(indicesArrayAttr[0]).getInt() != index))
        return failure();
    }
  }
  return failure();
}
