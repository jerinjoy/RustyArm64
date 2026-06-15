use crate::decode::Instruction;
use crate::memory::Memory;
use crate::registers::Registers;

/// Errors that can occur during instruction execution.
#[derive(Debug, PartialEq, Eq)]
pub enum ExecError {
    /// The HLT instruction was executed — the processor should stop.
    Halt,
    /// A memory access was out of bounds.
    MemoryFault(u64),
}

/// ARMv8-A AArch64 CPU state used by the execution engine.
pub struct Cpu {
    /// Architectural register file (X0–X30, SP, PC, flags).
    pub regs: Registers,
    /// Physical memory.
    pub mem: Memory,
    /// True when the processor has halted (HLT executed).
    pub halted: bool,
}

impl Cpu {
    /// Create a new CPU with the given memory, all registers zeroed,
    /// PC = 0, and `halted = false`.
    pub fn new(memory: Memory) -> Self {
        Self {
            regs: Registers::new(),
            mem: memory,
            halted: false,
        }
    }
}

/// Execute a single decoded instruction on the given CPU.
///
/// # Arguments
/// * `cpu` – mutable reference to the CPU state.
/// * `ins` – the decoded [`Instruction`].
///
/// # Returns
/// * `Ok(())` on normal completion.
/// * `Err(ExecError::Halt)` for HLT.
/// * `Err(ExecError::MemoryFault)` for out-of-bounds memory accesses.
pub fn execute_instruction(cpu: &mut Cpu, ins: Instruction) -> Result<(), ExecError> {
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
            let src = cpu.regs.read(rn);

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
            cpu.regs.write(rd, result);

            // ── Set flags if S=1 ─────────────────────────────────────
            if s {
                let msb = if sf { 63 } else { 31 };
                let n = (result >> msb) & 1 == 1;
                let z = result == 0;
                cpu.regs.set_flags(n, z, carry_out, overflow);
            }

            Ok(())
        }

        Instruction::MovWideImmediate {
            sf: _sf,
            opc: _opc,
            hw,
            imm16,
            rd,
        } => {
            let value = (imm16 as u64) << hw;
            cpu.regs.write(rd, value);
            Ok(())
        }

        Instruction::Hlt { imm16: _imm16 } => {
            cpu.halted = true;
            Err(ExecError::Halt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Instruction;

    /// Helper: create a CPU with 1 KiB of memory.
    fn new_cpu() -> Cpu {
        Cpu::new(Memory::new(1024))
    }

    // ── ADD immediate ─────────────────────────────────────────────────

    #[test]
    fn test_add_imm_64() {
        // ADD X0, X1, #5
        let mut cpu = new_cpu();
        cpu.regs.write(1, 10); // X1 = 10

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut cpu, ins).expect("should succeed");
        assert_eq!(cpu.regs.read(0), 15);
        // Flags unchanged (S=0).
        assert!(!cpu.regs.n);
        assert!(!cpu.regs.z);
        assert!(!cpu.regs.c);
        assert!(!cpu.regs.v);
    }

    #[test]
    fn test_sub_imm_64() {
        // SUB X2, X3, #4095, LSL #12
        let mut cpu = new_cpu();
        cpu.regs.write(3, 0x100_0000);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 2,
            rn: 3,
            imm12: 4095,
            shift: true,
        };

        execute_instruction(&mut cpu, ins).expect("should succeed");
        assert_eq!(cpu.regs.read(2), 0x1000);
    }

    #[test]
    fn test_add_imm_32() {
        // ADD W0, W1, #5  (32-bit, sf=0)
        let mut cpu = new_cpu();
        // Set X1 to a value with upper bits set; 32-bit op should ignore them.
        cpu.regs.write(1, 0xFFFF_FFFF_0000_000A); // lower 32 bits = 10

        let ins = Instruction::AddSubImmediate {
            sf: false,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut cpu, ins).expect("should succeed");
        // 32-bit result: (10 + 5) & 0xFFFF_FFFF = 15, zero-extended.
        assert_eq!(cpu.regs.read(0), 15);
    }

    // ── Zero-register / SP semantics ─────────────────────────────────

    #[test]
    fn test_add_sub_discard_when_rd_31() {
        // ADD XZR, X1, #5  (rd=31 → write discarded)
        let mut cpu = new_cpu();
        cpu.regs.write(1, 10);

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 31,
            rn: 1,
            imm12: 5,
            shift: false,
        };

        execute_instruction(&mut cpu, ins).expect("should succeed");
        // rd=31 writes are discarded — XZR reads back as 0.
        assert_eq!(cpu.regs.read(31), 0);
    }

    #[test]
    fn test_add_sub_rn_31_reads_zero() {
        // ADD X0, XZR, #100  (rn=31 → reads 0)
        let mut cpu = new_cpu();

        let ins = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 31,
            imm12: 100,
            shift: false,
        };

        execute_instruction(&mut cpu, ins).expect("should succeed");
        assert_eq!(cpu.regs.read(0), 100);
    }

    // ── Flags (NZCV) ─────────────────────────────────────────────────

    #[test]
    fn test_add_sub_flag_nzcv_set() {
        // ── Case 1: ADD with positive result, no carry, no overflow ──
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 10);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true, // S=1
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 15);
            assert!(!cpu.regs.n); // result positive
            assert!(!cpu.regs.z); // result not zero
            assert!(!cpu.regs.c); // no carry
            assert!(!cpu.regs.v); // no overflow
        }

        // ── Case 2: ADD that produces zero ───────────────────────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 0);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 0,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 0);
            assert!(!cpu.regs.n);
            assert!(cpu.regs.z); // Z=1
            assert!(!cpu.regs.c); // no carry (0 + 0 = 0)
            assert!(!cpu.regs.v);
        }

        // ── Case 3: ADD with carry out (unsigned overflow) ───────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 0xFFFF_FFFF_FFFF_FFFF); // max u64
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 0); // wraps to 0
            assert!(!cpu.regs.n);
            assert!(cpu.regs.z); // result = 0
            assert!(cpu.regs.c); // carry out
            assert!(!cpu.regs.v); // no signed overflow
        }

        // ── Case 4: ADD with signed overflow ─────────────────────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 0x7FFF_FFFF_FFFF_FFFF); // max positive i64
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 0x8000_0000_0000_0000); // negative
            assert!(cpu.regs.n); // negative
            assert!(!cpu.regs.z);
            assert!(!cpu.regs.c); // no carry
            assert!(cpu.regs.v); // signed overflow
        }

        // ── Case 5: SUB that sets carry (no borrow) ──────────────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 10);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true, // SUB
                s: true,
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 5);
            assert!(!cpu.regs.n);
            assert!(!cpu.regs.z);
            assert!(cpu.regs.c); // C=1 (no borrow: 10 >= 5)
            assert!(!cpu.regs.v);
        }

        // ── Case 6: SUB with borrow (C=0) ────────────────────────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 3);
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 5,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 0xFFFF_FFFF_FFFF_FFFEu64); // 3 - 5 = -2
            assert!(cpu.regs.n);
            assert!(!cpu.regs.z);
            assert!(!cpu.regs.c); // borrow: 3 < 5, so C=0
            assert!(!cpu.regs.v); // no signed overflow
        }

        // ── Case 7: SUB with signed overflow ─────────────────────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 0x8000_0000_0000_0000); // min i64
            let ins = Instruction::AddSubImmediate {
                sf: true,
                op: true,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 0x7FFF_FFFF_FFFF_FFFF); // positive
            assert!(!cpu.regs.n);
            assert!(!cpu.regs.z);
            assert!(cpu.regs.c); // 0x8000... >= 1 → no borrow
            assert!(cpu.regs.v); // signed overflow: negative - positive = positive
        }

        // ── Case 8: 32-bit addition, N reflects bit 31 ──────────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 0x7FFF_FFFF); // max positive i32
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            // 0x7FFF_FFFF + 1 = 0x8000_0000 (negative in 32-bit)
            assert_eq!(cpu.regs.read(0), 0x8000_0000);
            assert!(cpu.regs.n); // bit 31 set
            assert!(!cpu.regs.z);
            assert!(!cpu.regs.c); // no carry out of 32 bits
            assert!(cpu.regs.v); // signed overflow
        }

        // ── Case 9: 32-bit addition, C reflects bit 32 carry ────────
        {
            let mut cpu = new_cpu();
            cpu.regs.write(1, 0xFFFF_FFFF); // max u32
            let ins = Instruction::AddSubImmediate {
                sf: false,
                op: false,
                s: true,
                rd: 0,
                rn: 1,
                imm12: 1,
                shift: false,
            };
            execute_instruction(&mut cpu, ins).expect("should succeed");
            assert_eq!(cpu.regs.read(0), 0); // wraps to 0 in 32-bit
            assert!(!cpu.regs.n);
            assert!(cpu.regs.z);
            assert!(cpu.regs.c); // carry out of 32 bits
            assert!(!cpu.regs.v);
        }
    }

    // ── Encode-decode-execute round-trip ─────────────────────────────

    #[test]
    fn test_add_imm_via_encode() {
        let mut cpu = new_cpu();
        cpu.regs.write(1, 42);

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

        execute_instruction(&mut cpu, ins).expect("should succeed");
        assert_eq!(cpu.regs.read(0), 142);
    }

    #[test]
    fn test_sub_imm_shifted_via_encode() {
        let mut cpu = new_cpu();
        cpu.regs.write(5, 0x2000);

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

        execute_instruction(&mut cpu, ins).expect("should succeed");
        assert_eq!(cpu.regs.read(10), 0x1000);
    }
}
