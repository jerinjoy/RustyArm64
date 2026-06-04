# ARM64 MRA XML Schema — Tag Reference

Quick-reference guide to the XML elements used in the ARM Machine-Readable
Architecture specification. Use these to extract bit-field layouts and
execution pseudocode from the raw XML files.

---

## ISA XML files (`ISA_A64/ISA_A64_xml_A_profile-YYYY-MM/*.xml`)

Root element: `<instructionsection id="..." title="..." type="instruction">`

### Bit-field definitions

Located inside `<classes>` → `<iclass>` → `<regdiagram>`. Each `<box>` describes
a contiguous slice of the instruction encoding.

```xml
<regdiagram form="32" psname="...">
  <box hibit="31" width="1" name="sf" usename="1">
    <c colspan="1"/>               <!-- variable field -->
  </box>
  <box hibit="30" name="op" usename="1" settings="1" psbits="x">
    <c>0</c>                       <!-- constant 0 -->
  </box>
  <box hibit="28" width="6" settings="6">
    <c>1</c><c>0</c><c>0</c>...    <!-- 6-bit constant -->
  </box>
  <box hibit="21" width="12" name="imm12" usename="1">
    <c colspan="12"/>              <!-- 12-bit variable field -->
  </box>
  <box hibit="9" width="5" name="Rn" usename="1">
    <c colspan="5"/>               <!-- 5-bit register field -->
  </box>
  <box hibit="4" width="5" name="Rd" usename="1">
    <c colspan="5"/>               <!-- 5-bit register field -->
  </box>
</regdiagram>
```

**Key attributes on `<box>`:**

| Attribute | Meaning |
|-----------|---------|
| `hibit` | Highest bit index (31 = MSB of 32-bit instr) |
| `width` | Number of bits in this slice (default 1) |
| `name` | Field identifier used in pseudocode (e.g. "sf", "imm12", "Rn") |
| `usename="1"` | This field is **named** (variable); decode it |
| `settings="N"` | This field is **constant** (N fixed bits); verify at decode |
| `psbits` | Pseudocode bit pattern hint (e.g. "x" = don't-care) |

**How to compute bit ranges:**
- Slice spans `[hibit, hibit - width + 1]` (inclusive, MSB-indexed).
- Example: `hibit="21" width="12"` → bits `[21:10]`.
- Constant fields (no `name`) must be matched during decoding; they define the
  opcode bit-pattern. The `<c>` children give the expected value (MSB-first).

**Encoding variants** — `<encoding>` elements inside `<iclass>` describe 32-bit
vs 64-bit variants (toggled by the `sf` bit). Each `<encoding>` has its own
`<asmtemplate>` and may override `<box>` constants.

### Execution pseudocode

Two locations, distinguished by the `secttype` attribute on `<ps_section>`:

1. **Decode pseudocode** — inside `<iclass>`:
   ```xml
   <ps_section howmany="1">
     <ps name="..." sections="1" secttype="noheading">
       <pstext section="Decode" rep_section="decode">
         let d : integer{} = UInt(Rd);
         let datasize : integer{} = 32 &lt;&lt; UInt(sf);
         ...
       </pstext>
     </ps>
   </ps_section>
   ```
   This runs at **decode time**. It extracts operands from the bit-fields and
   validates encoding conditions. Failure → UNDEFINED.

2. **Execute pseudocode** — direct child of `<instructionsection>` (outside `<classes>`):
   ```xml
   <ps_section howmany="1">
     <ps name="..." sections="1" secttype="Operation">
       <pstext section="Execute" rep_section="execute">
         let operand1 : bits(datasize) = X{}(n);
         (result, -) = AddWithCarry(operand1, operand2, '0');
         X{datasize}(d) = result;
       </pstext>
     </ps>
   </ps_section>
   ```
   This runs at **execute time**. It performs the actual operation using the
   decoded operands.

**Identification rule:**
- `secttype="noheading"` → Decode
- `secttype="Operation"` → Execute

### Other useful elements

| Element | Location | Usage |
|---------|----------|-------|
| `<docvars>` / `<docvar>` | Top-level & inside `<iclass>` | `key="mnemonic"`, `key="instr-class"` |
| `<desc>` / `<brief>` | Top-level | Human-readable instruction summary |
| `<operationalnotes>` | Top-level | Special execution constraints |
| `<alias_list>` / `<aliasref>` | Top-level | Instruction aliases (e.g. MOV is alias for ADD) |
| `<explanations>` | After `<classes>` | Field-level descriptions, shift options, encoding tables |

---

## SysReg XML files (`SysReg/SysReg_xml_A_profile-YYYY-MM/*.xml`)

Root element: `<register_page>` → `<registers>` → `<register>`

### Register identity

```xml
<register execution_state="AArch64" is_register="True" is_internal="True">
  <reg_short_name>SCTLR_EL1</reg_short_name>
  <reg_long_name>System Control Register (EL1)</reg_long_name>
</register>
```

### Bit-field definitions

Inside `<reg_fieldsets>` → `<fields id="fieldset_0" length="64">` → `<field>`:

```xml
<field id="fieldset_0-63_63-1" reserved_type="RES0">
  <field_name>TIDCP</field_name>
  <field_msb>63</field_msb>
  <field_lsb>63</field_lsb>
  <field_description order="before">
    <para>Trap IMPLEMENTATION DEFINED functionality. ...</para>
  </field_description>
  <field_values impdef="False">
    <field_value_instance>
      <field_value>0b0</field_value>
      <field_value_description>...</field_value_description>
    </field_value_instance>
    <field_value_instance>
      <field_value>0b1</field_value>
      <field_value_description>...</field_value_description>
    </field_value_instance>
  </field_values>
  <fields_condition>When FEAT_TIDCP1 is implemented</fields_condition>
</field>
```

**Key elements on each `<field>`:**

| Element | Meaning |
|---------|---------|
| `<field_name>` | Bit-field name (e.g. "TIDCP") |
| `<field_msb>` / `<field_lsb>` | Bit range: `[MSB:LSB]` |
| `<field_description>` | Human-readable description |
| `<field_values>` | Enumeration of valid values with descriptions |
| `<field_access>` | Read/write access rules per exception level |
| `<field_resets>` | Reset value (Warm/Cold reset) |
| `<fields_condition>` | Feature flag gate (e.g. "When FEAT_xxx is implemented") |
| `reserved_type` | "RES0" (reserved, should be 0), "RES1" (reserved, should be 1), etc. |
| `rwtype` | Override access type for fallback variant |

**Field ID encoding:**
`fieldset_0-{MSB}_{LSB}-{variant}` — variant distinguishes alternative
definitions of the same bit range under different feature conditions.

**Multiple variants per bit range are common** (e.g. `fieldset_0-63_63-1` and
`fieldset_0-63_63-2` for the same bits under different feature flags). Always
read the `<fields_condition>` to know which variant applies.

### Additional register metadata

| Element | Meaning |
|---------|---------|
| `<reg_purpose>` | Functional description |
| `<reg_attributes>` | Width, endianness notes |
| `<reg_mappings>` | AArch32 ↔ AArch64 name mappings |
| `<reg_condition>` | Feature gate for the entire register |
