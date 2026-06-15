/// Error type returned when an instruction word cannot be decoded.
#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The instruction word does not match any known opcode pattern.
    UnknownOpcode(u32),
    /// The opcode pattern matches, but the encoding contains illegal field
    /// values.
    IllegalEncoding,
}

/// Represents a decoded ARMv8 AArch64 instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum Instruction {
    /// ADD or SUB (immediate).
    ///
    /// Fields:
    /// - `sf`  — size flag: `true` for 64‑bit, `false` for 32‑bit.
    /// - `op`  — operation: `false` for ADD, `true` for SUB.
    /// - `s`   — set-flags: `true` when the S (flags‑setting) bit is set.
    /// - `rd`  — destination register index (0‑31).
    /// - `rn`  — source register index (0‑31).
    /// - `imm12` — 12‑bit unsigned immediate.
    /// - `shift` — when `true`, `imm12` is left‑shifted by 12.
    AddSubImmediate {
        sf: bool,
        op: bool,
        s: bool,
        rd: u8,
        rn: u8,
        imm12: u16,
        shift: bool,
    },

    /// MOV (wide immediate): MOVZ, MOVN, or MOVK.
    ///
    /// Fields:
    /// - `sf`   — size flag: `true` for 64‑bit, `false` for 32‑bit.
    /// - `opc`  — opc field (bits [30:29]): 0 = MOVN, 2 = MOVZ, 3 = MOVK.
    /// - `hw`   — left‑shift amount (0, 16, 32, or 48).
    /// - `imm16` — 16‑bit immediate.
    /// - `rd`   — destination register index (0‑31).
    MovWideImmediate {
        sf: bool,
        opc: u8,
        hw: u8,
        imm16: u16,
        rd: u8,
    },

    /// HLT (Halt).
    ///
    /// Fixed encoding pattern: `1101010100000011` in bits [31:16].
    /// `imm16` is the 16‑bit immediate from bits [15:0].
    Hlt {
        imm16: u16,
    },
}

/// Decode a 32-bit little-endian AArch64 instruction word.
///
/// Returns `Ok(Instruction)` on success, or `Err(DecodeError)` if the word
/// does not match any recognised encoding.
///
/// # Encodings recognised
///
/// | Mnemonic             | Bits checked                                   |
/// |----------------------|------------------------------------------------|
/// | ADD/SUB (imm)        | `bits[28:24] == 0b10001`                       |
/// | MOV wide immediate   | `bits[28:23] == 0b100101`                      |
/// | HLT                  | `bits[31:16] == 0xD403`                        |
pub fn decode(word: u32) -> Result<Instruction, DecodeError> {
    // ── HLT ────────────────────────────────────────────────────────────
    // Pattern: 1101010100000011 in bits [31:16], imm16 in bits [15:0].
    if word >> 16 == 0xD403 {
        let imm16 = (word & 0xFFFF) as u16;
        return Ok(Instruction::Hlt { imm16 });
    }

    // ── ADD / SUB (immediate) ─────────────────────────────────────────
    // Encoding:  sf | op | S | 1 0 0 0 1 | sh | imm12 | Rn | Rd
    // Bits [28:24] == 0b10001.
    let opcode_28_24 = (word >> 24) & 0x1F;
    if opcode_28_24 == 0x11 {
        let sf = (word >> 31) & 1 == 1;
        let op = (word >> 30) & 1 == 1;
        let s = (word >> 29) & 1 == 1;
        let shift = (word >> 22) & 0x3 == 1; // 0b01 = LSL #12
        let imm12 = ((word >> 10) & 0xFFF) as u16;
        let rn = ((word >> 5) & 0x1F) as u8;
        let rd = (word & 0x1F) as u8;
        return Ok(Instruction::AddSubImmediate {
            sf,
            op,
            s,
            rd,
            rn,
            imm12,
            shift,
        });
    }

    // ── MOV wide immediate ────────────────────────────────────────────
    // Encoding:  sf | opc[1:0] | 1 0 0 1 0 1 | hw | imm16 | Rd
    // Bits [28:23] == 0b100101.
    let opcode_28_23 = (word >> 23) & 0x3F;
    if opcode_28_23 == 0b10_0101 {
        let sf = (word >> 31) & 1 == 1;
        let opc = ((word >> 29) & 0x3) as u8;
        let hw = (((word >> 21) & 0x3) * 16) as u8;
        let imm16 = ((word >> 5) & 0xFFFF) as u16;
        let rd = (word & 0x1F) as u8;
        return Ok(Instruction::MovWideImmediate {
            sf,
            opc,
            hw,
            imm16,
            rd,
        });
    }

    // ── Unknown ───────────────────────────────────────────────────────
    Err(DecodeError::UnknownOpcode(word))
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
        let inst = decode(0x9100_1420).expect("should decode ADD X0, X1, #5");
        let expected = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: false,
            rd: 0,
            rn: 1,
            imm12: 5,
            shift: false,
        };
        assert_instruction_eq(&inst, &expected, "ADD X0, X1, #5");
    }

    #[test]
    fn test_decode_add_imm_32() {
        // ADD W5, W6, #42  →  0x1100A8C5  (sf=0)
        // sf=0, op=0, S=0, sh=0, imm12=42, rn=6, rd=5
        let inst = decode(0x1100_A8C5).expect("should decode ADD W5, W6, #42");
        let expected = Instruction::AddSubImmediate {
            sf: false,
            op: false,
            s: false,
            rd: 5,
            rn: 6,
            imm12: 42,
            shift: false,
        };
        assert_instruction_eq(&inst, &expected, "ADD W5, W6, #42");
    }

    // ── SUB immediate tests ───────────────────────────────────────────

    #[test]
    fn test_decode_sub_imm_shift() {
        // SUB X2, X3, #4095, LSL #12  →  0xD17FFC62
        // sf=1, op=1, S=0, sh=1, imm12=4095, rn=3, rd=2
        let inst = decode(0xD17F_FC62).expect("should decode SUB X2, X3, #4095, LSL #12");
        let expected = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 2,
            rn: 3,
            imm12: 4095,
            shift: true,
        };
        assert_instruction_eq(&inst, &expected, "SUB X2, X3, #4095, LSL #12");
    }

    #[test]
    fn test_decode_sub_imm_no_shift() {
        // SUB X10, X11, #1  →  0xD100056A
        // sf=1, op=1, S=0, sh=0, imm12=1, rn=11, rd=10
        let inst = decode(0xD100_056A).expect("should decode SUB X10, X11, #1");
        let expected = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: false,
            rd: 10,
            rn: 11,
            imm12: 1,
            shift: false,
        };
        assert_instruction_eq(&inst, &expected, "SUB X10, X11, #1");
    }

    // ── MOVZ tests ────────────────────────────────────────────────────

    #[test]
    fn test_decode_movz_hw16() {
        // MOVZ X4, #0xABCD, LSL #16  →  0xD2B579A4
        //
        // Encoding:  sf=1 | opc=10 | 100101 | hw=01 | imm16=0xABCD | rd=4
        //   Word:  1101 0010 1011 0101 0111 1001 1010 0100
        let inst = decode(0xD2B5_79A4).expect("should decode MOVZ X4, #0xABCD, LSL #16");
        let expected = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 16,
            imm16: 0xABCD,
            rd: 4,
        };
        assert_instruction_eq(&inst, &expected, "MOVZ X4, #0xABCD, LSL #16");
    }

    #[test]
    fn test_decode_movz_hw0() {
        // MOVZ X0, #0x42, LSL #0  →  0xD2800840
        let inst = decode(0xD280_0840).expect("should decode MOVZ X0, #0x42");
        let expected = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 0,
            imm16: 0x42,
            rd: 0,
        };
        assert_instruction_eq(&inst, &expected, "MOVZ X0, #0x42");
    }

    #[test]
    fn test_decode_movz_hw48() {
        // MOVZ X5, #0x1, LSL #48  →  0xD2E00025
        let inst = decode(0xD2E0_0025).expect("should decode MOVZ X5, #1, LSL #48");
        let expected = Instruction::MovWideImmediate {
            sf: true,
            opc: 2,
            hw: 48,
            imm16: 1,
            rd: 5,
        };
        assert_instruction_eq(&inst, &expected, "MOVZ X5, #1, LSL #48");
    }

    // ── MOVK tests ────────────────────────────────────────────────────

    #[test]
    fn test_decode_movk() {
        // MOVK X7, #0xBEEF, LSL #32  →  0xF2D7DDE7
        //
        // Encoding:  sf=1 | opc=11 | 100101 | hw=10 | imm16=0xBEEF | rd=7
        //   hw = 2 → shift = 32
        //   Word: 1111_0010_1101_0111_1101_1101_1110_0111
        let inst = decode(0xF2D7_DDE7).expect("should decode MOVK X7, #0xBEEF, LSL #32");
        let expected = Instruction::MovWideImmediate {
            sf: true,
            opc: 3,
            hw: 32,
            imm16: 0xBEEF,
            rd: 7,
        };
        assert_instruction_eq(&inst, &expected, "MOVK X7, #0xBEEF, LSL #32");
    }

    // ── HLT tests ─────────────────────────────────────────────────────

    #[test]
    fn test_decode_hlt_0() {
        // HLT #0  →  0xD403_0000
        let inst = decode(0xD403_0000).expect("should decode HLT #0");
        let expected = Instruction::Hlt { imm16: 0 };
        assert_instruction_eq(&inst, &expected, "HLT #0");
    }

    #[test]
    fn test_decode_hlt_imm() {
        // HLT #42  →  0xD403_002A  (imm16 = 42 = 0x002A)
        let inst = decode(0xD403_002A).expect("should decode HLT #42");
        let expected = Instruction::Hlt { imm16: 42 };
        assert_instruction_eq(&inst, &expected, "HLT #42");
    }

    // ── Unknown instruction tests ─────────────────────────────────────

    #[test]
    fn test_decode_unknown_all_zero() {
        let result = decode(0x0000_0000);
        assert_eq!(result, Err(DecodeError::UnknownOpcode(0x0000_0000)));
    }

    #[test]
    fn test_decode_unknown_random() {
        let result = decode(0xDEAD_BEEF);
        assert_eq!(result, Err(DecodeError::UnknownOpcode(0xDEAD_BEEF)));
    }

    #[test]
    fn test_decode_unknown_near_miss_hlt() {
        // Bits [31:16] == 0xD402 — one bit off from 0xD403.
        let result = decode(0xD402_0000);
        assert_eq!(result, Err(DecodeError::UnknownOpcode(0xD402_0000)));
    }

    #[test]
    fn test_decode_unknown_near_miss_movz() {
        // Bits [28:23] == 0b101000 — one bit off from 0b100101 (bit 25 = 1).
        // This is NOT a valid wide-immediate encoding and won't match ADD/SUB
        // (bits[28:24] == 0b10100 ≠ 0b10001) or HLT (bits[31:16] == 0xD220 ≠
        // 0xD403).
        // Word: 0xD220_0020 = 1101_0010_0010_0000_0000_0000_0010_0000
        let result = decode(0xD220_0020);
        assert_eq!(result, Err(DecodeError::UnknownOpcode(0xD220_0020)));
    }

    // ── Edge-case tests ───────────────────────────────────────────────

    #[test]
    fn test_decode_add_s_imm() {
        // ADDS X7, X8, #0  →  0xB1000107  (S=1, flags-setting variant)
        let inst = decode(0xB100_0107).expect("should decode ADDS X7, X8, #0");
        let expected = Instruction::AddSubImmediate {
            sf: true,
            op: false,
            s: true,
            rd: 7,
            rn: 8,
            imm12: 0,
            shift: false,
        };
        assert_instruction_eq(&inst, &expected, "ADDS X7, X8, #0");
    }

    #[test]
    fn test_decode_sub_s_imm() {
        // SUBS X9, X10, #100  →  0xF1019149  (S=1)
        let inst = decode(0xF101_9149).expect("should decode SUBS X9, X10, #100");
        let expected = Instruction::AddSubImmediate {
            sf: true,
            op: true,
            s: true,
            rd: 9,
            rn: 10,
            imm12: 100,
            shift: false,
        };
        assert_instruction_eq(&inst, &expected, "SUBS X9, X10, #100");
    }

    #[test]
    fn test_decode_invalid_word() {
        // A word that matches no known pattern returns UnknownOpcode.
        // 0xFFFF_FFFF should match nothing.
        let result = decode(0xFFFF_FFFF);
        assert_eq!(result, Err(DecodeError::UnknownOpcode(0xFFFF_FFFF)));
    }
}
