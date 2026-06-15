/// Represents a decoded ARMv8 AArch64 instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum Instruction {
    /// ADD or SUB (immediate), non-flags-setting variant.
    ///
    /// `sf` — size flag: `true` for 64‑bit, `false` for 32‑bit.
    /// `op` — operation: `false` for ADD, `true` for SUB.
    /// `shift` — when `true`, the 12‑bit immediate is left-shifted by 12.
    /// `imm12` — 12‑bit unsigned immediate (zero‑extended for use).
    /// `rn` — source register index (0‑31).
    /// `rd` — destination register index (0‑31).
    AddSub {
        sf: bool,
        op: bool,
        shift: bool,
        imm12: u16,
        rn: u8,
        rd: u8,
    },
    /// MOVZ (Move wide with zero).
    ///
    /// Sets `rd` to `imm16 << hw`, zeroing all other bits.
    /// `hw` is the left-shift amount and may only be 0, 16, 32, or 48.
    Movz {
        hw: u8,
        imm16: u16,
        rd: u8,
    },
    /// HLT (Halt).
    ///
    /// Recognised when bits [31:24] == 0xD4 and bits [4:0] == 0.
    /// Any 16‑bit immediate in bits [20:5] is accepted.
    Hlt,
    /// Any instruction not recognised by this decoder.
    Unknown,
}

/// Decode a 32-bit little-endian AArch64 instruction word.
///
/// # Correctness
///
/// The decoder matches the following encodings:
///
/// - **AddSub** — bits [28:24] == `10001`.
///   - `sf` = bit 31
///   - `op` = bit 30
///   - `shift` = 1 when bits [23:22] == `01` (LSL #12)
///   - `imm12` = bits [21:10]
///   - `rn` = bits [9:5]
///   - `rd` = bits [4:0]
///
/// - **Movz** — bits [30:29] == `10` and bits [28:23] == `100101`.
///   - `hw` = bits [22:21] × 16
///   - `imm16` = bits [20:5]
///   - `rd` = bits [4:0]
///
/// - **Hlt** — bits [31:24] == `0xD4` and bits [4:0] == `0`.
///
/// Any word that does not match the above is returned as `Unknown`.
pub fn decode(inst: u32) -> Instruction {
    // ── HLT ────────────────────────────────────────────────────────────
    // HLT: bits [31:24] == 0xD4, bits [4:0] == 0.
    if (inst >> 24) == 0xD4 && (inst & 0x1F) == 0 {
        return Instruction::Hlt;
    }

    // ── ADD / SUB (immediate) ─────────────────────────────────────────
    // Encoding:  sf | op | S | 1 0 0 0 1 | sh | imm12 | Rn | Rd
    // Bits [28:24] == 0b10001.
    let opcode_28_24 = (inst >> 24) & 0x1F;
    if opcode_28_24 == 0x11 {
        let sf = (inst >> 31) & 1 == 1;
        let op = (inst >> 30) & 1 == 1;
        let shift = (inst >> 22) & 0x3 == 1; // 0b01 = LSL #12
        let imm12 = ((inst >> 10) & 0xFFF) as u16;
        let rn = ((inst >> 5) & 0x1F) as u8;
        let rd = (inst & 0x1F) as u8;
        return Instruction::AddSub {
            sf,
            op,
            shift,
            imm12,
            rn,
            rd,
        };
    }

    // ── MOVZ ──────────────────────────────────────────────────────────
    // Encoding:  sf | 1 0 | 1 0 0 1 0 1 | hw | imm16 | Rd
    // Decode only MOVZ: bits [30:29] == 0b10, bits [28:23] == 0b100101.
    let opc_30_29 = (inst >> 29) & 0x3;
    let opc_28_23 = (inst >> 23) & 0x3F;
    if opc_30_29 == 0b10 && opc_28_23 == 0b10_0101 {
        let hw = (((inst >> 21) & 0x3) * 16) as u8;
        let imm16 = ((inst >> 5) & 0xFFFF) as u16;
        let rd = (inst & 0x1F) as u8;
        return Instruction::Movz { hw, imm16, rd };
    }

    // ── Unknown ───────────────────────────────────────────────────────
    Instruction::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert that two instructions are equal, with a descriptive
    /// message on failure.
    fn assert_instruction_eq(got: &Instruction, expected: &Instruction, msg: &str) {
        assert!(
            got == expected,
            "{}: expected {:?}, got {:?}",
            msg,
            expected,
            got
        );
    }

    // ── ADD immediate tests ───────────────────────────────────────────

    #[test]
    fn test_decode_add_imm_64() {
        // ADD X0, X1, #5  →  0x91001420
        // sf=1, op=0, S=0, sh=0, imm12=5, rn=1, rd=0
        let inst = decode(0x9100_1420);
        let expected = Instruction::AddSub {
            sf: true,
            op: false,
            shift: false,
            imm12: 5,
            rn: 1,
            rd: 0,
        };
        assert_instruction_eq(&inst, &expected, "ADD X0, X1, #5");
    }

    #[test]
    fn test_decode_add_imm_32() {
        // ADD W5, W6, #42  →  0x1100A8C5  (sf=0)
        // sf=0, op=0, S=0, sh=0, imm12=42, rn=6, rd=5
        let inst = decode(0x1100_A8C5);
        let expected = Instruction::AddSub {
            sf: false,
            op: false,
            shift: false,
            imm12: 42,
            rn: 6,
            rd: 5,
        };
        assert_instruction_eq(&inst, &expected, "ADD W5, W6, #42");
    }

    // ── SUB immediate tests ───────────────────────────────────────────

    #[test]
    fn test_decode_sub_imm_shift() {
        // SUB X2, X3, #4095, LSL #12  →  0xD17FFC62
        // sf=1, op=1, S=0, sh=1, imm12=4095, rn=3, rd=2
        let inst = decode(0xD17F_FC62);
        let expected = Instruction::AddSub {
            sf: true,
            op: true,
            shift: true,
            imm12: 4095,
            rn: 3,
            rd: 2,
        };
        assert_instruction_eq(&inst, &expected, "SUB X2, X3, #4095, LSL #12");
    }

    #[test]
    fn test_decode_sub_imm_no_shift() {
        // SUB X10, X11, #1  →  0xD100056A
        // sf=1, op=1, S=0, sh=0, imm12=1, rn=11, rd=10
        let inst = decode(0xD100_056A);
        let expected = Instruction::AddSub {
            sf: true,
            op: true,
            shift: false,
            imm12: 1,
            rn: 11,
            rd: 10,
        };
        assert_instruction_eq(&inst, &expected, "SUB X10, X11, #1");
    }

    // ── MOVZ tests ────────────────────────────────────────────────────

    #[test]
    fn test_decode_movz_hw16() {
        // MOVZ X4, #0xABCD, LSL #16  →  0xD2B579A4
        //
        // Encoding:  sf=1 | 10 | 100101 | hw=01 | imm16=0xABCD | rd=4
        //   imm16 = 0xABCD = 1010_1011_1100_1101
        //   Word:  1101 0010 1011 0101 0111 1001 1010 0100
        let inst = decode(0xD2B5_79A4);
        let expected = Instruction::Movz {
            hw: 16,
            imm16: 0xABCD,
            rd: 4,
        };
        assert_instruction_eq(&inst, &expected, "MOVZ X4, #0xABCD, LSL #16");
    }

    #[test]
    fn test_decode_movz_hw0() {
        // MOVZ X0, #0x42, LSL #0  →  0xD2800840
        let inst = decode(0xD280_0840);
        let expected = Instruction::Movz {
            hw: 0,
            imm16: 0x42,
            rd: 0,
        };
        assert_instruction_eq(&inst, &expected, "MOVZ X0, #0x42");
    }

    #[test]
    fn test_decode_movz_hw48() {
        // MOVZ X5, #0x1, LSL #48  →  0xD2E00025
        let inst = decode(0xD2E0_0025);
        let expected = Instruction::Movz {
            hw: 48,
            imm16: 1,
            rd: 5,
        };
        assert_instruction_eq(&inst, &expected, "MOVZ X5, #1, LSL #48");
    }

    // ── HLT tests ─────────────────────────────────────────────────────

    #[test]
    fn test_decode_hlt_0() {
        // HLT #0  →  0xD4400000
        let inst = decode(0xD440_0000);
        assert_instruction_eq(&inst, &Instruction::Hlt, "HLT #0");
    }

    #[test]
    fn test_decode_hlt_imm() {
        // HLT #42  →  0xD4400540  (imm16=42 in bits [20:5])
        let inst = decode(0xD440_0540);
        assert_instruction_eq(&inst, &Instruction::Hlt, "HLT #42");
    }

    // ── Unknown instruction tests ─────────────────────────────────────

    #[test]
    fn test_decode_unknown_all_zero() {
        let inst = decode(0x0000_0000);
        assert_instruction_eq(&inst, &Instruction::Unknown, "all zeroes");
    }

    #[test]
    fn test_decode_unknown_random() {
        let inst = decode(0xDEAD_BEEF);
        assert_instruction_eq(&inst, &Instruction::Unknown, "0xDEADBEEF");
    }

    #[test]
    fn test_decode_unknown_near_miss_hlt() {
        // Bits [31:24] == 0xD4 but bits [4:0] != 0 — not valid HLT.
        let inst = decode(0xD440_0001);
        let expected = Instruction::Unknown;
        assert_instruction_eq(&inst, &expected, "HLT-like with non-zero rd");
    }

    #[test]
    fn test_decode_unknown_near_miss_movz() {
        // MOVN (bits [30:29] = 00) should not decode as MOVZ.
        // MOVN X0, #1  →  0x92800020
        let inst = decode(0x9280_0020);
        assert_instruction_eq(&inst, &Instruction::Unknown, "MOVN (not MOVZ)");
    }

    // ── Edge-case tests ───────────────────────────────────────────────

    #[test]
    fn test_decode_add_s_imm() {
        // ADDS X7, X8, #0  →  0xB1000107  (S=1, flags-setting variant)
        // Our decoder should still accept it as AddSub.
        let inst = decode(0xB100_0107);
        let expected = Instruction::AddSub {
            sf: true,
            op: false,
            shift: false,
            imm12: 0,
            rn: 8,
            rd: 7,
        };
        assert_instruction_eq(&inst, &expected, "ADDS X7, X8, #0");
    }

    #[test]
    fn test_decode_sub_s_imm() {
        // SUBS X9, X10, #100  →  0xF1019149  (S=1)
        let inst = decode(0xF101_9149);
        let expected = Instruction::AddSub {
            sf: true,
            op: true,
            shift: false,
            imm12: 100,
            rn: 10,
            rd: 9,
        };
        assert_instruction_eq(&inst, &expected, "SUBS X9, X10, #100");
    }
}
