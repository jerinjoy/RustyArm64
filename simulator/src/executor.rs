use crate::decode::Instruction;
use crate::memory::PhysicalMemory;
use crate::registers::Registers;

/// Errors that can occur during instruction execution.
#[derive(Debug, PartialEq, Eq)]
pub enum ExecError {
    /// A memory access was out of bounds.
    MemoryFault(u64),
}

/// Execute a single decoded instruction on the given CPU state.
///
/// # Arguments
/// * `regs` – mutable reference to the architectural register file.
/// * `_mem` – mutable reference to physical memory (reserved for
///   future load/store instructions).
/// * `halted` – mutable reference to the halt flag; set to `true`
///   when HLT executes.
/// * `ins` – the decoded [`Instruction`].
///
/// # Returns
/// * `Ok(())` on normal completion, including HLT.
/// * `Err(ExecError::MemoryFault)` for out-of-bounds memory accesses.
pub fn execute_instruction(
    regs: &mut Registers,
    _mem: &mut PhysicalMemory,
    halted: &mut bool,
    ins: Instruction,
) -> Result<(), ExecError> {
    match ins {
        Instruction::AddSubImmediate {
            sf,
            op,
            s,
            rd,
            rn,
            imm12,
            shift,
        } => {
            // ── Compute the immediate operand ─────────────────────────
            let operand: u64 = (imm12 as u64) << if shift { 12 } else { 0 };

            // ── Read source register (rn=31 → zero register) ─────────
            let src = regs.read(rn);

            // ── Perform add or sub ───────────────────────────────────
            let (result, carry_out, overflow) = if op {
                // SUB: result = src - operand
                let res = src.wrapping_sub(operand);
                // Carry (C) is NOT Borrow: set to 1 if src >= operand (unsigned).
                let c = src >= operand;
                // Overflow for subtraction: sign change when src and operand
                // have different signs and result has sign different from src.
                let msb = if sf { 63 } else { 31 };
                let src_sign = (src >> msb) & 1;
                let op_sign = (operand >> msb) & 1;
                let res_sign = (res >> msb) & 1;
                let v = (src_sign != op_sign) && (res_sign != src_sign);
                (res, c, v)
            } else {
                // ADD: result = src + operand
                let res = src.wrapping_add(operand);
                // Carry: unsigned overflow out of the relevant width.
                let c = if sf {
                    src.checked_add(operand).is_none()
                } else {
                    let src32 = src as u32;
                    let op32 = operand as u32;
                    src32.checked_add(op32).is_none()
                };
                // Overflow for addition: sign change when both operands
                // have the same sign but result has a different sign.
                let msb = if sf { 63 } else { 31 };
                let src_sign = (src >> msb) & 1;
                let op_sign = (operand >> msb) & 1;
                let res_sign = (res >> msb) & 1;
                let v = (src_sign == op_sign) && (res_sign != src_sign);
                (res, c, v)
            };

            // ── 32-bit operation: zero-extend result ─────────────────
            let result = if sf { result } else { result & 0xFFFF_FFFFu64 };

            // ── Write destination (rd=31 → zero register, discarded) ─
            regs.write(rd, result);

            // ── Set flags if S=1 ─────────────────────────────────────
            if s {
                let msb = if sf { 63 } else { 31 };
                let n = (result >> msb) & 1 == 1;
                let z = result == 0;
                regs.set_flags(n, z, carry_out, overflow);
            }

            Ok(())
        }

        Instruction::MovWideImmediate {
            sf,
            opc,
            hw,
            imm16,
            rd,
        } => {
            // ── Build the value according to opc ─────────────────────
            let value = match opc {
                0b10 => {
                    // MOVZ: clear all bits, then insert imm16 at hw.
                    (imm16 as u64) << hw
                }
                0b11 => {
                    // MOVK: keep existing register value, insert
                    // imm16 into the 16-bit field at position hw.
                    let existing = regs.read(rd);
                    let mask = !(0xFFFFu64 << hw);
                    (existing & mask) | ((imm16 as u64) << hw)
                }
                0b00 => {
                    // MOVN: bitwise-NOT of (imm16 shifted by hw).
                    !((imm16 as u64) << hw)
                }
                _ => {
                    // Reserved opc — treat as no-op.
                    return Ok(());
                }
            };

            // ── 32-bit operation: zero-extend ────────────────────────
            let value = if sf { value } else { value & 0xFFFF_FFFF };

            // ── Write destination (rd=31 → zero register, discarded) ─
            regs.write(rd, value);

            Ok(())
        }

        Instruction::Hlt { imm16: _imm16 } => {
            *halted = true;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Instruction;

    /// Helper: create a fresh pair of registers and memory (1 KiB)
    /// plus a halt flag.
    fn setup() -> (Registers, PhysicalMemory, bool) {
        (Registers::new(), PhysicalMemory::new(), false)
    }

    // ── HLT ──────────────────────────────────────────────────────────

    #[test]
    fn test_hlt_sets_halted_flag() {
        let (mut regs, mut mem, mut halted) = setup();
        assert!(!halted);

        let ins = Instruction::Hlt { imm16: 0 };
        let result = execute_instruction(&mut regs, &mut mem, &mut halted, ins);

        assert_eq!(result, Ok(()));
        assert!(halted);
    }

    #[test]
    fn test_hlt_non_zero_imm16() {
        // HLT with a non-zero imm16 should still set the halted flag.
        let (mut regs, mut mem, mut halted) = setup();
        assert!(!halted);

        let ins = Instruction::Hlt { imm16: 42 };
        let result = execute_instruction(&mut regs, &mut mem, &mut halted, ins);

        assert_eq!(result, Ok(()));
        assert!(halted);
    }

    // ── ADD immediate ─────────────────────────────────────────────────

    #[test]
    fn test_add_imm_64() {
        // ADD X0, X1, #5
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 10); // X1 = 10

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(0), 15);
        // Flags unchanged (S=0).
        assert!(!regs.n());
        assert!(!regs.z());
        assert!(!regs.c());
        assert!(!regs.v());
    }

    #[test]
    fn test_add_imm_64_shifted() {
        // ADD X0, X1, #1, LSL #12  → operand = 0x1000
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 0x2000);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 1,
            shift: true,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(0), 0x3000);
    }

    #[test]
    fn test_add_imm_64_max_shifted() {
        // ADD X0, X1, #4095, LSL #12  → operand = 0xFFF000
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 0x1000);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 4095,
            shift: true,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 0x1000 + 0xFFF000 = 0x1000000
        assert_eq!(regs.read(0), 0x100_0000);
    }

    #[test]
    fn test_add_imm_32() {
        // ADD W0, W1, #5  (32-bit, sf=0)
        let (mut regs, mut mem, mut halted) = setup();
        // Set X1 to a value with upper bits set; 32-bit op should ignore them.
        regs.write(1, 0xFFFF_FFFF_0000_000A); // lower 32 bits = 10

        let ins = Instruction::AddSubImmediate {
            sf: false,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 32-bit result: (10 + 5) & 0xFFFF_FFFF = 15, zero-extended.
        assert_eq!(regs.read(0), 15);
    }

    #[test]
    fn test_add_imm_32_shifted() {
        // ADD W0, W1, #1, LSL #12  (32-bit, shifted)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 0xFFFF_FFFF_0000_2000); // lower 32 = 0x2000

        let ins = Instruction::AddSubImmediate {
            sf: false,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 1,
            shift: true,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 32-bit: (0x2000 + 0x1000) & 0xFFFF_FFFF = 0x3000
        assert_eq!(regs.read(0), 0x3000);
    }

    // ── SUB immediate ─────────────────────────────────────────────────

    #[test]
    fn test_sub_imm_64() {
        // SUB X2, X3, #4095, LSL #12
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(3, 0x100_0000);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 2,
            rn: 3,
            imm12: 4095,
            shift: true,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(2), 0x1000);
    }

    #[test]
    fn test_sub_imm_64_no_shift() {
        // SUB X10, X11, #42  (no shift)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(11, 100);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 10,
            rn: 11,
            imm12: 42,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(10), 58);
    }

    #[test]
    fn test_sub_imm_32() {
        // SUB W0, W1, #5  (32-bit)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 0xFFFF_FFFF_0000_000A); // lower 32 = 10

        let ins = Instruction::AddSubImmediate {
            sf: false,
            op: true,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 32-bit: 10 - 5 = 5
        assert_eq!(regs.read(0), 5);
    }

    #[test]
    fn test_sub_imm_32_shifted() {
        // SUB W0, W1, #1, LSL #12  (32-bit, shifted)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 0xFFFF_FFFF_0000_2000); // lower 32 = 0x2000

        let ins = Instruction::AddSubImmediate {
            sf: false,
            op: true,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 1,
            shift: true,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 32-bit: (0x2000 - 0x1000) & 0xFFFF_FFFF = 0x1000
        assert_eq!(regs.read(0), 0x1000);
    }

    // ── Zero-register / SP semantics ─────────────────────────────────

    #[test]
    fn test_add_sub_discard_when_rd_31() {
        // ADD XZR, X1, #5  (rd=31 → write discarded)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 10);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 31,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // rd=31 writes are discarded — XZR reads back as 0.
        assert_eq!(regs.read(31), 0);
    }

    #[test]
    fn test_add_sub_rn_31_reads_zero() {
        // ADD X0, XZR, #100  (rn=31 → reads 0)
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 31,
            imm12: 100,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(0), 100);
    }

    #[test]
    fn test_sub_discard_when_rd_31() {
        // SUB XZR, X1, #5  (rd=31, write discarded)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 10);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 31,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(31), 0);
    }

    #[test]
    fn test_sub_rn_31_reads_zero() {
        // SUB X0, XZR, #5  (rn=31 reads 0, so 0 - 5 = -5)
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 0,
            rn: 31,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 0 - 5 = -5 (0xFFFF_FFFF_FFFF_FFFB)
        assert_eq!(regs.read(0), 0xFFFF_FFFF_FFFF_FFFB);
    }

    // ── Flags (NZCV) ─────────────────────────────────────────────────

    #[test]
    fn test_add_sub_flag_nzcv_set() {
        // ── Case 1: ADD with positive result, no carry, no overflow ──
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 10);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true, // S=1
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 15);
            assert!(!regs.n()); // result positive
            assert!(!regs.z()); // result not zero
            assert!(!regs.c()); // no carry
            assert!(!regs.v()); // no overflow
        }

        // ── Case 2: ADD that produces zero ───────────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 0,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0);
            assert!(!regs.n());
            assert!(regs.z()); // Z=1
            assert!(!regs.c()); // no carry (0 + 0 = 0)
            assert!(!regs.v());
        }

        // ── Case 3: ADD with carry out (unsigned overflow) ───────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0xFFFF_FFFF_FFFF_FFFF); // max u64
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0); // wraps to 0
            assert!(!regs.n());
            assert!(regs.z()); // result = 0
            assert!(regs.c()); // carry out
            assert!(!regs.v()); // no signed overflow
        }

        // ── Case 4: ADD with signed overflow ─────────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0x7FFF_FFFF_FFFF_FFFF); // max positive i64
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0x8000_0000_0000_0000); // negative
            assert!(regs.n()); // negative
            assert!(!regs.z());
            assert!(!regs.c()); // no carry
            assert!(regs.v()); // signed overflow
        }

        // ── Case 5: SUB that sets carry (no borrow) ──────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 10);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true, // SUB
                s: true,
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 5);
            assert!(!regs.n());
            assert!(!regs.z());
            assert!(regs.c()); // C=1 (no borrow: 10 >= 5)
            assert!(!regs.v());
        }

        // ── Case 6: SUB with borrow (C=0) ────────────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 3);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0xFFFF_FFFF_FFFF_FFFEu64); // 3 - 5 = -2
            assert!(regs.n());
            assert!(!regs.z());
            assert!(!regs.c()); // borrow: 3 < 5, so C=0
            assert!(!regs.v()); // no signed overflow
        }

        // ── Case 7: SUB with signed overflow ─────────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0x8000_0000_0000_0000); // min i64
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0x7FFF_FFFF_FFFF_FFFF); // positive
            assert!(!regs.n());
            assert!(!regs.z());
            assert!(regs.c()); // 0x8000... >= 1 → no borrow
            assert!(regs.v()); // signed overflow: negative - positive = positive
        }

        // ── Case 8: 32-bit addition, N reflects bit 31 ──────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0x7FFF_FFFF); // max positive i32
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            // 0x7FFF_FFFF + 1 = 0x8000_0000 (negative in 32-bit)
            assert_eq!(regs.read(0), 0x8000_0000);
            assert!(regs.n()); // bit 31 set
            assert!(!regs.z());
            assert!(!regs.c()); // no carry out of 32 bits
            assert!(regs.v()); // signed overflow
        }

        // ── Case 9: 32-bit addition, C reflects bit 32 carry ────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0xFFFF_FFFF); // max u32
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0); // wraps to 0 in 32-bit
            assert!(!regs.n());
            assert!(regs.z());
            assert!(regs.c()); // carry out of 32 bits
            assert!(!regs.v());
        }

        // ── Case 10: 32-bit SUB, C reflects no borrow ────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 10);
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 5);
            assert!(!regs.n());
            assert!(!regs.z());
            assert!(regs.c()); // C=1: no borrow
            assert!(!regs.v());
        }

        // ── Case 11: 32-bit SUB with borrow ──────────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 1);
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            // 1 - 5 = -4 → 0xFFFF_FFFC in 32-bit, zero-extended to 64-bit
            assert_eq!(regs.read(0), 0xFFFF_FFFC);
            assert!(regs.n()); // negative
            assert!(!regs.z());
            assert!(!regs.c()); // borrow: 1 < 5
            assert!(!regs.v()); // no signed overflow
        }

        // ── Case 12: 32-bit SUB with signed overflow ─────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0x8000_0000); // min i32
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            // 0x8000_0000 - 1 = 0x7FFF_FFFF (positive in 32-bit)
            assert_eq!(regs.read(0), 0x7FFF_FFFF);
            assert!(!regs.n()); // positive
            assert!(!regs.z());
            assert!(regs.c()); // 0x8000_0000 >= 1 → no borrow
            assert!(regs.v()); // signed overflow
        }

        // ── Case 13: ADD shifted with S=1 ────────────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0x7FFF_FFFF_FFFF_F000);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: true, // operand = 0x1000
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            // 0x7FFF_FFFF_FFFF_F000 + 0x1000 = 0x8000_0000_0000_0000
            assert_eq!(regs.read(0), 0x8000_0000_0000_0000);
            assert!(regs.n()); // negative
            assert!(!regs.z());
            assert!(!regs.c()); // no carry
            assert!(regs.v()); // signed overflow
        }

        // ── Case 14: SUB shifted with S=1, carry set ─────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.write(1, 0x2000);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: true, // operand = 0x1000
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0x1000);
            assert!(!regs.n());
            assert!(!regs.z());
            assert!(regs.c()); // 0x2000 >= 0x1000
            assert!(!regs.v());
        }

        // ── Case 15: S=0 should not modify flags ─────────────────────
        {
            let (mut regs, mut mem, mut halted) = setup();
            // Set some initial flags.
            regs.set_flags(true, true, true, true);
            regs.write(1, 0);

            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: false, // S=0: flags should be preserved
                rd: 0,
                rn: 1,
                imm12: 0,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            // Flags should remain unchanged.
            assert!(regs.n());
            assert!(regs.z());
            assert!(regs.c());
            assert!(regs.v());
        }

        // ── Case 16: ADD zero + zero with S=1 sets Z, clears N,C,V ──
        {
            let (mut regs, mut mem, mut halted) = setup();
            regs.set_flags(true, false, true, true); // pre-set flags
            regs.write(1, 0);

            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 0,
                shift: false,
            };
            execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
            assert_eq!(regs.read(0), 0);
            assert!(!regs.n());
            assert!(regs.z()); // zero result
            assert!(!regs.c()); // no carry from 0+0
            assert!(!regs.v()); // no overflow
        }
    }

    // ── Encode-decode-execute round-trip ─────────────────────────────

    #[test]
    fn test_add_imm_via_encode() {
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(1, 42);

        // ADD X0, X1, #100  (no shift)
        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 100,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(0), 142);
    }

    #[test]
    fn test_sub_imm_shifted_via_encode() {
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(5, 0x2000);

        // SUB X10, X5, #1, LSL #12  → subtract 0x1000
        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 10,
            rn: 5,
            imm12: 1,
            shift: true,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(10), 0x1000);
    }

    #[test]
    fn test_add_imm_32_via_encode() {
        // ADD W15, W20, #7  (32-bit)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(20, 100);

        let ins = Instruction::AddSubImmediate {
            sf: false,
            op: false,
            s: false,
            rd: 15,
            rn: 20,
            imm12: 7,
            shift: false,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(15), 107);
    }

    // ── MOV wide immediate ───────────────────────────────────────────

    #[test]
    fn test_movz_64_shift0() {
        // MOVZ X3, #0xBEEF, LSL #0  → opc=2, hw=0, imm16=0xBEEF, rd=3
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 0,
            imm16: 0xBEEF,
            rd: 3,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(3), 0xBEEF);
    }

    #[test]
    fn test_movz_64_shift16() {
        // MOVZ X4, #0xABCD, LSL #16
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 16,
            imm16: 0xABCD,
            rd: 4,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(4), 0xABCD_0000);
    }

    #[test]
    fn test_movz_64_shift32() {
        // MOVZ X6, #0x1234, LSL #32
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 32,
            imm16: 0x1234,
            rd: 6,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(6), 0x1234_0000_0000);
    }

    #[test]
    fn test_movz_64_shift48() {
        // MOVZ X7, #0xABCD, LSL #48  → opc=2, hw=48, imm16=0xABCD, rd=7
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 48,
            imm16: 0xABCD,
            rd: 7,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(7), 0xABCD_0000_0000_0000);
    }

    #[test]
    fn test_movk_preserve_other_bits() {
        // Set up X5 with a known pattern, then MOVK into bits [47:32].
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(5, 0xDEAD_0000_BEEF_CAFE);

        // MOVK X5, #0x1234, LSL #32  → opc=3, hw=32, imm16=0x1234, rd=5
        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 3,
            hw: 32,
            imm16: 0x1234,
            rd: 5,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // Bits [47:32] become 0x1234; all other 16-bit fields stay intact.
        assert_eq!(regs.read(5), 0xDEAD_1234_BEEF_CAFE);
    }

    #[test]
    fn test_movk_32() {
        // MOVK W8, #0xF00D, LSL #16  (32-bit variant)
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(8, 0xAAAA_BBBB_CCCC_DDDD); // junk in upper bits

        let ins = Instruction::MovWideImmediate {
            sf: false,
            opc: 3,
            hw: 16,
            imm16: 0xF00D,
            rd: 8,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // 32-bit MOVK: result = (existing & ~0xFFFF0000) | (0xF00D << 16)
        // existing = 0xCCCC_DDDD (lower 32 of original)
        // mask clears bits [31:16]: 0xCCCC_DDDD & 0x0000_FFFF = 0x0000_DDDD
        // insert: 0xF00D << 16 = 0xF00D_0000
        // result = 0xF00D_DDDD, zero-extended
        assert_eq!(regs.read(8), 0xF00D_DDDD);
    }

    #[test]
    fn test_movk_shift0() {
        // MOVK X9, #0xBEEF, LSL #0 (insert at bits [15:0])
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(9, 0xDEAD_CAFE_1234_5678);

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 3,
            hw: 0,
            imm16: 0xBEEF,
            rd: 9,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(9), 0xDEAD_CAFE_1234_BEEF);
    }

    #[test]
    fn test_movk_shift48() {
        // MOVK X10, #0xDEAD, LSL #48
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(10, 0xBEEF_CAFE_1234_5678);

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 3,
            hw: 48,
            imm16: 0xDEAD,
            rd: 10,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(10), 0xDEAD_CAFE_1234_5678);
    }

    #[test]
    fn test_movz_32_zero_extend() {
        // MOVZ W2, #0xFFFF, LSL #16  → sf=0, opc=2, hw=16, imm16=0xFFFF, rd=2
        // 32-bit operation must zero the upper 32 bits of the register.
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(2, 0xAAAA_BBBB_CCCC_DDDD); // pre-fill with junk

        let ins = Instruction::MovWideImmediate {
            sf: false,
            opc: 2,
            hw: 16,
            imm16: 0xFFFF,
            rd: 2,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // Result: [31:16] = 0xFFFF, [15:0] = 0x0000, upper 32 bits = 0.
        assert_eq!(regs.read(2), 0xFFFF_0000);
    }

    #[test]
    fn test_mov_wide_discard_when_rd_31() {
        // MOVZ XZR, #0x42, LSL #0 → rd=31, write is discarded.
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 0,
            imm16: 0x42,
            rd: 31,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        // XZR always reads back as 0.
        assert_eq!(regs.read(31), 0);
    }

    // ── MOVN / reserved opc ──────────────────────────────────────────

    #[test]
    fn test_movn_64_shift0() {
        // MOVN X5, #0x42  →  ~0x42 = 0xFFFF_FFFF_FFFF_FFBD
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 0,
            hw: 0,
            imm16: 0x42,
            rd: 5,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(5), !0x42u64);
    }

    #[test]
    fn test_movn_64_shift16() {
        // MOVN X0, #1, LSL #16  →  ~0x0001_0000 = 0xFFFF_FFFF_FFFE_FFFF
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 0,
            hw: 16,
            imm16: 1,
            rd: 0,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(0), !(1u64 << 16));
    }

    #[test]
    fn test_movn_32() {
        // MOVN W3, #0  →  ~0 in 32-bit = 0xFFFF_FFFF (zero-extended to 64-bit)
        let (mut regs, mut mem, mut halted) = setup();

        let ins = Instruction::MovWideImmediate {
            sf: false,
            opc: 0,
            hw: 0,
            imm16: 0,
            rd: 3,
        };

        execute_instruction(&mut regs, &mut mem, &mut halted, ins).expect("should succeed");
        assert_eq!(regs.read(3), 0xFFFF_FFFF);
    }

    #[test]
    fn test_mov_wide_reserved_opc1_no_op() {
        // Reserved opc=1 is treated as no-op.
        let (mut regs, mut mem, mut halted) = setup();
        regs.write(5, 0xCAFE_F00D);

        let ins = Instruction::MovWideImmediate {
            sf: true,
            opc: 1, // reserved
            hw: 16,
            imm16: 0xBEEF,
            rd: 5,
        };

        let result = execute_instruction(&mut regs, &mut mem, &mut halted, ins);
        assert_eq!(result, Ok(()));
        // Register should be unchanged.
        assert_eq!(regs.read(5), 0xCAFE_F00D);
        assert!(!halted);
    }

    // ── ExecError tests ──────────────────────────────────────────────

    #[test]
    fn test_exec_error_debug_eq() {
        let e1 = ExecError::MemoryFault(0x1000);
        let e2 = ExecError::MemoryFault(0x1000);
        let e3 = ExecError::MemoryFault(0x2000);

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
        assert_eq!(format!("{:?}", e1), "MemoryFault(4096)");
    }
}
