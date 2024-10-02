//===- DeduceVersionExtensionCapabilityPass.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to deduce minimal version/extension/capability
// requirements for a spirv::ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

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

#if 0

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
class GlobalVariableOp;
} // namespace spirv
} // namespace mlir

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
// ::mlir::spirv::GlobalVariableOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GlobalVariableOpGenericAdaptorBase {
public:
  struct Properties {
    using bindingTy = ::mlir::IntegerAttr;
    bindingTy binding;

    auto getBinding() {
      auto &propStorage = this->binding;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setBinding(const ::mlir::IntegerAttr &propValue) {
      this->binding = propValue;
    }
    using builtinTy = ::mlir::StringAttr;
    builtinTy builtin;

    auto getBuiltin() {
      auto &propStorage = this->builtin;
      return ::llvm::dyn_cast_or_null<::mlir::StringAttr>(propStorage);
    }
    void setBuiltin(const ::mlir::StringAttr &propValue) {
      this->builtin = propValue;
    }
    using descriptor_setTy = ::mlir::IntegerAttr;
    descriptor_setTy descriptor_set;

    auto getDescriptorSet() {
      auto &propStorage = this->descriptor_set;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setDescriptorSet(const ::mlir::IntegerAttr &propValue) {
      this->descriptor_set = propValue;
    }
    using initializerTy = ::mlir::FlatSymbolRefAttr;
    initializerTy initializer;

    auto getInitializer() {
      auto &propStorage = this->initializer;
      return ::llvm::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(propStorage);
    }
    void setInitializer(const ::mlir::FlatSymbolRefAttr &propValue) {
      this->initializer = propValue;
    }
    using linkage_attributesTy = ::mlir::spirv::LinkageAttributesAttr;
    linkage_attributesTy linkage_attributes;

    auto getLinkageAttributes() {
      auto &propStorage = this->linkage_attributes;
      return ::llvm::dyn_cast_or_null<::mlir::spirv::LinkageAttributesAttr>(propStorage);
    }
    void setLinkageAttributes(const ::mlir::spirv::LinkageAttributesAttr &propValue) {
      this->linkage_attributes = propValue;
    }
    using locationTy = ::mlir::IntegerAttr;
    locationTy location;

    auto getLocation() {
      auto &propStorage = this->location;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setLocation(const ::mlir::IntegerAttr &propValue) {
      this->location = propValue;
    }
    using sym_nameTy = ::mlir::StringAttr;
    sym_nameTy sym_name;

    auto getSymName() {
      auto &propStorage = this->sym_name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setSymName(const ::mlir::StringAttr &propValue) {
      this->sym_name = propValue;
    }
    using typeTy = ::mlir::TypeAttr;
    typeTy type;

    auto getType() {
      auto &propStorage = this->type;
      return ::llvm::cast<::mlir::TypeAttr>(propStorage);
    }
    void setType(const ::mlir::TypeAttr &propValue) {
      this->type = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.binding == this->binding &&
        rhs.builtin == this->builtin &&
        rhs.descriptor_set == this->descriptor_set &&
        rhs.initializer == this->initializer &&
        rhs.linkage_attributes == this->linkage_attributes &&
        rhs.location == this->location &&
        rhs.sym_name == this->sym_name &&
        rhs.type == this->type &&
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
  GlobalVariableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("spirv.GlobalVariable", odsAttrs.getContext());
  }

  GlobalVariableOpGenericAdaptorBase(GlobalVariableOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::TypeAttr getTypeAttr() {
    auto attr = ::llvm::cast<::mlir::TypeAttr>(getProperties().type);
    return attr;
  }

  ::mlir::Type getType();
  ::mlir::StringAttr getSymNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().sym_name);
    return attr;
  }

  ::llvm::StringRef getSymName();
  ::mlir::FlatSymbolRefAttr getInitializerAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(getProperties().initializer);
    return attr;
  }

  ::std::optional< ::llvm::StringRef > getInitializer();
  ::mlir::IntegerAttr getLocationAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().location);
    return attr;
  }

  ::std::optional<uint32_t> getLocation();
  ::mlir::IntegerAttr getBindingAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().binding);
    return attr;
  }

  ::std::optional<uint32_t> getBinding();
  ::mlir::IntegerAttr getDescriptorSetAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().descriptor_set);
    return attr;
  }

  ::std::optional<uint32_t> getDescriptorSet();
  ::mlir::StringAttr getBuiltinAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().builtin);
    return attr;
  }

  ::std::optional< ::llvm::StringRef > getBuiltin();
  ::mlir::spirv::LinkageAttributesAttr getLinkageAttributesAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::spirv::LinkageAttributesAttr>(getProperties().linkage_attributes);
    return attr;
  }

  ::std::optional<::mlir::spirv::LinkageAttributesAttr> getLinkageAttributes();
};
} // namespace detail
template <typename RangeT>
class GlobalVariableOpGenericAdaptor : public detail::GlobalVariableOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GlobalVariableOpGenericAdaptorBase;
public:
  GlobalVariableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GlobalVariableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GlobalVariableOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  GlobalVariableOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : GlobalVariableOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = GlobalVariableOp, typename = std::enable_if_t<std::is_same_v<LateInst, GlobalVariableOp>>>
  GlobalVariableOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class GlobalVariableOpAdaptor : public GlobalVariableOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GlobalVariableOpGenericAdaptor::GlobalVariableOpGenericAdaptor;
  GlobalVariableOpAdaptor(GlobalVariableOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GlobalVariableOp : public ::mlir::Op<GlobalVariableOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::SymbolOpInterface::Trait, ::mlir::spirv::QueryMinVersionInterface::Trait, ::mlir::spirv::QueryMaxVersionInterface::Trait, ::mlir::spirv::QueryExtensionInterface::Trait, ::mlir::spirv::QueryCapabilityInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GlobalVariableOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GlobalVariableOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("binding"), ::llvm::StringRef("builtin"), ::llvm::StringRef("descriptor_set"), ::llvm::StringRef("initializer"), ::llvm::StringRef("linkage_attributes"), ::llvm::StringRef("location"), ::llvm::StringRef("sym_name"), ::llvm::StringRef("type")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBindingAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBindingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBuiltinAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBuiltinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDescriptorSetAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDescriptorSetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getInitializerAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getInitializerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLinkageAttributesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLinkageAttributesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getLocationAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getLocationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getTypeAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("spirv.GlobalVariable");
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
  ::mlir::TypeAttr getTypeAttr() {
    return ::llvm::cast<::mlir::TypeAttr>(getProperties().type);
  }

  ::mlir::Type getType();
  ::mlir::StringAttr getSymNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().sym_name);
  }

  ::llvm::StringRef getSymName();
  ::mlir::FlatSymbolRefAttr getInitializerAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(getProperties().initializer);
  }

  ::std::optional< ::llvm::StringRef > getInitializer();
  ::mlir::IntegerAttr getLocationAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().location);
  }

  ::std::optional<uint32_t> getLocation();
  ::mlir::IntegerAttr getBindingAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().binding);
  }

  ::std::optional<uint32_t> getBinding();
  ::mlir::IntegerAttr getDescriptorSetAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().descriptor_set);
  }

  ::std::optional<uint32_t> getDescriptorSet();
  ::mlir::StringAttr getBuiltinAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::StringAttr>(getProperties().builtin);
  }

  ::std::optional< ::llvm::StringRef > getBuiltin();
  ::mlir::spirv::LinkageAttributesAttr getLinkageAttributesAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::spirv::LinkageAttributesAttr>(getProperties().linkage_attributes);
  }

  ::std::optional<::mlir::spirv::LinkageAttributesAttr> getLinkageAttributes();
  void setTypeAttr(::mlir::TypeAttr attr) {
    getProperties().type = attr;
  }

  void setType(::mlir::Type attrValue);
  void setSymNameAttr(::mlir::StringAttr attr) {
    getProperties().sym_name = attr;
  }

  void setSymName(::llvm::StringRef attrValue);
  void setInitializerAttr(::mlir::FlatSymbolRefAttr attr) {
    getProperties().initializer = attr;
  }

  void setInitializer(::std::optional<::llvm::StringRef> attrValue);
  void setLocationAttr(::mlir::IntegerAttr attr) {
    getProperties().location = attr;
  }

  void setLocation(::std::optional<uint32_t> attrValue);
  void setBindingAttr(::mlir::IntegerAttr attr) {
    getProperties().binding = attr;
  }

  void setBinding(::std::optional<uint32_t> attrValue);
  void setDescriptorSetAttr(::mlir::IntegerAttr attr) {
    getProperties().descriptor_set = attr;
  }

  void setDescriptorSet(::std::optional<uint32_t> attrValue);
  void setBuiltinAttr(::mlir::StringAttr attr) {
    getProperties().builtin = attr;
  }

  void setBuiltin(::std::optional<::llvm::StringRef> attrValue);
  void setLinkageAttributesAttr(::mlir::spirv::LinkageAttributesAttr attr) {
    getProperties().linkage_attributes = attr;
  }

  ::mlir::Attribute removeInitializerAttr() {
      auto &attr = getProperties().initializer;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeLocationAttr() {
      auto &attr = getProperties().location;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeBindingAttr() {
      auto &attr = getProperties().binding;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeDescriptorSetAttr() {
      auto &attr = getProperties().descriptor_set;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeBuiltinAttr() {
      auto &attr = getProperties().builtin;
      attr = {};
      return attr;
  }

  ::mlir::Attribute removeLinkageAttributesAttr() {
      auto &attr = getProperties().linkage_attributes;
      attr = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeAttr type, StringAttr sym_name, FlatSymbolRefAttr initializer = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeAttr type, ArrayRef<NamedAttribute> namedAttrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, StringRef name, unsigned descriptorSet, unsigned binding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, StringRef name, spirv::BuiltIn builtin);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, StringRef sym_name, FlatSymbolRefAttr initializer = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeAttr type, ::mlir::StringAttr sym_name, /*optional*/::mlir::FlatSymbolRefAttr initializer, /*optional*/::mlir::IntegerAttr location, /*optional*/::mlir::IntegerAttr binding, /*optional*/::mlir::IntegerAttr descriptor_set, /*optional*/::mlir::StringAttr builtin, /*optional*/::mlir::spirv::LinkageAttributesAttr linkage_attributes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr type, ::mlir::StringAttr sym_name, /*optional*/::mlir::FlatSymbolRefAttr initializer, /*optional*/::mlir::IntegerAttr location, /*optional*/::mlir::IntegerAttr binding, /*optional*/::mlir::IntegerAttr descriptor_set, /*optional*/::mlir::StringAttr builtin, /*optional*/::mlir::spirv::LinkageAttributesAttr linkage_attributes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type type, ::llvm::StringRef sym_name, /*optional*/::mlir::FlatSymbolRefAttr initializer, /*optional*/::mlir::IntegerAttr location, /*optional*/::mlir::IntegerAttr binding, /*optional*/::mlir::IntegerAttr descriptor_set, /*optional*/::mlir::StringAttr builtin, /*optional*/::mlir::spirv::LinkageAttributesAttr linkage_attributes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type type, ::llvm::StringRef sym_name, /*optional*/::mlir::FlatSymbolRefAttr initializer, /*optional*/::mlir::IntegerAttr location, /*optional*/::mlir::IntegerAttr binding, /*optional*/::mlir::IntegerAttr descriptor_set, /*optional*/::mlir::StringAttr builtin, /*optional*/::mlir::spirv::LinkageAttributesAttr linkage_attributes);
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
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  ::mlir::spirv::StorageClass storageClass() {
    return ::llvm::cast<::mlir::spirv::PointerType>(this->getType()).getStorageClass();
  }
};
} // namespace spirv
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::GlobalVariableOp)


#endif

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVUPDATEVCEPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

namespace {
/// Pass to deduce minimal version/extension/capability requirements for a
/// spirv::ModuleOp.
class UpdateVCEPass final
    : public spirv::impl::SPIRVUpdateVCEPassBase<UpdateVCEPass> {
  void runOnOperation() override;
};
} // namespace

/// Checks that `candidates` extension requirements are possible to be satisfied
/// with the given `targetEnv` and updates `deducedExtensions` if so. Emits
/// errors attaching to the given `op` on failures.
///
///  `candidates` is a vector of vector for extension requirements following
/// ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
/// convention.
static LogicalResult checkAndUpdateExtensionRequirements(
    Operation *op, const spirv::TargetEnv &targetEnv,
    const spirv::SPIRVType::ExtensionArrayRefVector &candidates,
    SetVector<spirv::Extension> &deducedExtensions) {
  for (const auto &ors : candidates) {
    if (std::optional<spirv::Extension> chosen = targetEnv.allows(ors)) {
      deducedExtensions.insert(*chosen);
    } else {
      SmallVector<StringRef, 4> extStrings;
      for (spirv::Extension ext : ors)
        extStrings.push_back(spirv::stringifyExtension(ext));

      return op->emitError("'")
             << op->getName() << "' requires at least one extension in ["
             << llvm::join(extStrings, ", ")
             << "] but none allowed in target environment";
    }
  }
  return success();
}

/// Checks that `candidates`capability requirements are possible to be satisfied
/// with the given `targetEnv` and updates `deducedCapabilities` if so. Emits
/// errors attaching to the given `op` on failures.
///
///  `candidates` is a vector of vector for capability requirements following
/// ((Capability::A OR Capability::B) AND (Capability::C OR Capability::D))
/// convention.
static LogicalResult checkAndUpdateCapabilityRequirements(
    Operation *op, const spirv::TargetEnv &targetEnv,
    const spirv::SPIRVType::CapabilityArrayRefVector &candidates,
    SetVector<spirv::Capability> &deducedCapabilities) {
  for (const auto &ors : candidates) {
    if (std::optional<spirv::Capability> chosen = targetEnv.allows(ors)) {
      deducedCapabilities.insert(*chosen);
    } else {
      SmallVector<StringRef, 4> capStrings;
      for (spirv::Capability cap : ors)
        capStrings.push_back(spirv::stringifyCapability(cap));

      return op->emitError("'")
             << op->getName() << "' requires at least one capability in ["
             << llvm::join(capStrings, ", ")
             << "] but none allowed in target environment";
    }
  }
  return success();
}

void UpdateVCEPass::runOnOperation() {
  spirv::ModuleOp module = getOperation();

  spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnv(module);
  if (!targetAttr) {
    module.emitError("missing 'spirv.target_env' attribute");
    return signalPassFailure();
  }

  spirv::TargetEnv targetEnv(targetAttr);
  spirv::Version allowedVersion = targetAttr.getVersion();

  spirv::Version deducedVersion = spirv::Version::V_1_0;
  SetVector<spirv::Extension> deducedExtensions;
  SetVector<spirv::Capability> deducedCapabilities;

  // Walk each SPIR-V op to deduce the minimal version/extension/capability
  // requirements.
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    // Op min version requirements
    if (auto minVersionIfx = dyn_cast<spirv::QueryMinVersionInterface>(op)) {
      std::optional<spirv::Version> minVersion = minVersionIfx.getMinVersion();
      if (minVersion) {
        deducedVersion = std::max(deducedVersion, *minVersion);
        if (deducedVersion > allowedVersion) {
          return op->emitError("'")
                 << op->getName() << "' requires min version "
                 << spirv::stringifyVersion(deducedVersion)
                 << " but target environment allows up to "
                 << spirv::stringifyVersion(allowedVersion);
        }
      }
    }

    // Op extension requirements
    if (auto extensions = dyn_cast<spirv::QueryExtensionInterface>(op))
      if (failed(checkAndUpdateExtensionRequirements(
              op, targetEnv, extensions.getExtensions(), deducedExtensions)))
        return WalkResult::interrupt();

    // Op capability requirements
    if (auto capabilities = dyn_cast<spirv::QueryCapabilityInterface>(op))
      if (failed(checkAndUpdateCapabilityRequirements(
              op, targetEnv, capabilities.getCapabilities(),
              deducedCapabilities)))
        return WalkResult::interrupt();

    SmallVector<Type, 4> valueTypes;
    valueTypes.append(op->operand_type_begin(), op->operand_type_end());
    valueTypes.append(op->result_type_begin(), op->result_type_end());

    // Special treatment for global variables, whose type requirements are
    // conveyed by type attributes.
    if (auto globalVar = dyn_cast<spirv::GlobalVariableOp>(op))
      valueTypes.push_back(globalVar.getType());

    // Requirements from values' types
    SmallVector<ArrayRef<spirv::Extension>, 4> typeExtensions;
    SmallVector<ArrayRef<spirv::Capability>, 8> typeCapabilities;
    for (Type valueType : valueTypes) {
      typeExtensions.clear();
      cast<spirv::SPIRVType>(valueType).getExtensions(typeExtensions);
      if (failed(checkAndUpdateExtensionRequirements(
              op, targetEnv, typeExtensions, deducedExtensions)))
        return WalkResult::interrupt();

      typeCapabilities.clear();
      cast<spirv::SPIRVType>(valueType).getCapabilities(typeCapabilities);
      if (failed(checkAndUpdateCapabilityRequirements(
              op, targetEnv, typeCapabilities, deducedCapabilities)))
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return signalPassFailure();

  // TODO: verify that the deduced version is consistent with
  // SPIR-V ops' maximal version requirements.

  auto triple = spirv::VerCapExtAttr::get(
      deducedVersion, deducedCapabilities.getArrayRef(),
      deducedExtensions.getArrayRef(), &getContext());
  module->setAttr(spirv::ModuleOp::getVCETripleAttrName(), triple);
}
