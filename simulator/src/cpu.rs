use crate::decode::{self, Instruction};

/// Errors that can occur during CPU execution.
#[derive(Debug, PartialEq, Eq)]
pub enum SimError {
    /// The program executed a HLT instruction.
    Halt,
    /// An unrecognized instruction word was encountered.
    UnknownInstruction(u32),
    /// Memory access at the given address was out of bounds.
    MemoryFault(u64),
}

/// Represents the state of an ARMv8 CPU.
pub struct Cpu {
    /// General-purpose registers X0-X30 (64-bit).
    /// X31 is the stack pointer (SP), stored separately.
    regs: [u64; 31],
    /// Stack Pointer (X31 / SP).
    sp: u64,
    /// Program Counter.
    pc: u64,
    /// Processor State Register (PSTATE).
    pstate: u64,
}

impl Cpu {
    /// Create a new CPU with all registers zeroed and PC = 0.
    pub fn new() -> Self {
        Self {
            regs: [0u64; 31],
            sp: 0,
            pc: 0,
            pstate: 0,
        }
    }

    /// Read a general-purpose register.
    ///
    /// # Panics
    /// Panics if `index` is greater than 30.
    pub fn read_reg(&self, index: usize) -> u64 {
        assert!(index <= 30, "register index {} out of range (0..=30)", index);
        self.regs[index]
    }

    /// Write a general-purpose register (X0-X30).
    ///
    /// # Panics
    /// Panics if `index` is greater than 30.
    pub fn write_reg(&mut self, index: usize, value: u64) {
        assert!(index <= 30, "register index {} out of range (0..=30)", index);
        self.regs[index] = value;
    }

    /// Read the stack pointer (SP).
    pub fn read_sp(&self) -> u64 {
        self.sp
    }

    /// Write the stack pointer (SP).
    pub fn write_sp(&mut self, value: u64) {
        self.sp = value;
    }

    /// Read the program counter.
    pub fn read_pc(&self) -> u64 {
        self.pc
    }

    /// Write the program counter.
    pub fn write_pc(&mut self, value: u64) {
        self.pc = value;
    }

    /// Read the PSTATE register.
    pub fn read_pstate(&self) -> u64 {
        self.pstate
    }

    /// Write the PSTATE register.
    pub fn write_pstate(&mut self, value: u64) {
        self.pstate = value;
    }

    // ── Instruction execution ────────────────────────────────────────

    /// Fetch a 32-bit instruction from memory at the current PC.
    ///
    /// Reads 4 bytes in little-endian order. Does not advance PC.
    /// Returns `Err(SimError::MemoryFault)` if the read goes out of bounds.
    pub fn fetch(&self, mem: &[u8]) -> Result<u32, SimError> {
        let pc = self.pc as usize;
        if pc + 4 > mem.len() {
            return Err(SimError::MemoryFault(self.pc));
        }
        let b0 = mem[pc] as u32;
        let b1 = mem[pc + 1] as u32;
        let b2 = mem[pc + 2] as u32;
        let b3 = mem[pc + 3] as u32;
        Ok(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    /// Execute a decoded instruction, reading/writing registers and memory.
    ///
    /// Returns `Ok(())` on success, or `Err(SimError::Halt)` for HLT.
    /// Unknown instructions are handled before calling this method (see
    /// [`step`](Self::step)).
    pub fn execute(
        &mut self,
        _mem: &mut [u8],
        inst: Instruction,
    ) -> Result<(), SimError> {
        match inst {
            Instruction::AddSub {
                sf,
                op,
                shift,
                imm12,
                rn,
                rd,
            } => {
                // Compute the immediate value.
                let imm: u64 = if shift {
                    (imm12 as u64) << 12
                } else {
                    imm12 as u64
                };

                // Read source register.  Rn == 31 aliases SP.
                let rn_val = if rn == 31 {
                    self.sp
                } else {
                    self.read_reg(rn as usize)
                };

                // Perform the operation (ADD or SUB).
                let mut result = if op {
                    rn_val.wrapping_sub(imm)
                } else {
                    rn_val.wrapping_add(imm)
                };

                // For 32-bit operations (sf == false), mask to 32 bits.
                if !sf {
                    result &= 0xFFFF_FFFF;
                }

                // Write destination.  Rd == 31 aliases SP.
                if rd != 31 {
                    self.write_reg(rd as usize, result);
                } else {
                    self.sp = result;
                }

                Ok(())
            }
            Instruction::Movz { hw, imm16, rd } => {
                // MOVZ sets rd to (imm16 << hw), zeroing all other bits.
                let value = (imm16 as u64) << hw;
                if rd != 31 {
                    self.write_reg(rd as usize, value);
                } else {
                    self.sp = value;
                }
                Ok(())
            }
            Instruction::Hlt => Err(SimError::Halt),
            Instruction::Unknown => {
                // This case is handled by `step` before calling `execute`.
                // If it is reached, something went wrong.
                Err(SimError::UnknownInstruction(0))
            }
        }
    }

    /// Fetch, decode, and execute a single instruction (a "step").
    ///
    /// On success, advances PC by 4.
    /// On `HLT`, returns `Err(SimError::Halt)` without modifying PC.
    /// On an unknown instruction, returns
    /// `Err(SimError::UnknownInstruction)` without modifying PC.
    /// On a memory fault during fetch, returns
    /// `Err(SimError::MemoryFault)` without modifying PC.
    pub fn step(&mut self, mem: &mut [u8]) -> Result<(), SimError> {
        let inst_word = self.fetch(mem)?;

        // Handle unknown instructions before execution so we can
        // include the raw instruction word in the error.
        let inst = decode::decode(inst_word);
        if matches!(inst, Instruction::Unknown) {
            return Err(SimError::UnknownInstruction(inst_word));
        }

        match self.execute(mem, inst) {
            Err(SimError::Halt) => Err(SimError::Halt),
            Ok(()) => {
                self.pc = self.pc.wrapping_add(4);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing CPU tests ───────────────────────────────────────────

    #[test]
    fn test_new_cpu_all_zero() {
        let cpu = Cpu::new();
        for i in 0..=30 {
            assert_eq!(cpu.read_reg(i), 0, "register X{} should be 0", i);
        }
        assert_eq!(cpu.read_sp(), 0);
        assert_eq!(cpu.read_pc(), 0);
        assert_eq!(cpu.read_pstate(), 0);
    }

    #[test]
    fn test_write_and_read_reg() {
        let mut cpu = Cpu::new();
        cpu.write_reg(0, 0xDEAD_BEEF);
        assert_eq!(cpu.read_reg(0), 0xDEAD_BEEF);

        cpu.write_reg(30, 0xCAFE);
        assert_eq!(cpu.read_reg(30), 0xCAFE);
    }

    #[test]
    fn test_write_and_read_sp() {
        let mut cpu = Cpu::new();
        cpu.write_sp(0xFFFF_0000);
        assert_eq!(cpu.read_sp(), 0xFFFF_0000);
    }

    #[test]
    fn test_write_and_read_pc() {
        let mut cpu = Cpu::new();
        cpu.write_pc(0x8000);
        assert_eq!(cpu.read_pc(), 0x8000);
    }

    #[test]
    fn test_write_and_read_pstate() {
        let mut cpu = Cpu::new();
        cpu.write_pstate(0x3C0); // EL1h, interrupts masked
        assert_eq!(cpu.read_pstate(), 0x3C0);
    }

    #[test]
    fn test_default_trait() {
        let cpu: Cpu = Default::default();
        assert_eq!(cpu.read_pc(), 0);
    }

    #[test]
    #[should_panic(expected = "register index 31 out of range")]
    fn test_read_reg_out_of_range() {
        let cpu = Cpu::new();
        cpu.read_reg(31);
    }

    #[test]
    #[should_panic(expected = "register index 31 out of range")]
    fn test_write_reg_out_of_range() {
        let mut cpu = Cpu::new();
        cpu.write_reg(31, 0);
    }

    // ── Helper: store a 32-bit word into a &mut [u8] at `addr` ───────

    fn write_inst(mem: &mut [u8], addr: u64, word: u32) {
        let a = addr as usize;
        mem[a] = word as u8;
        mem[a + 1] = (word >> 8) as u8;
        mem[a + 2] = (word >> 16) as u8;
        mem[a + 3] = (word >> 24) as u8;
    }

    // ── Step tests ────────────────────────────────────────────────────

    #[test]
    fn test_step_add() {
        // ADD X0, X1, #5  →  0x91001420
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x9100_1420);

        let mut cpu = Cpu::new();
        cpu.write_reg(1, 10); // X1 = 10
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert!(result.is_ok(), "step should succeed");

        // X0 = 10 + 5 = 15
        assert_eq!(cpu.read_reg(0), 15);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_step_sub() {
        // SUB X2, X3, #4095, LSL #12  →  0xD17FFC62
        // 4095 << 12 = 0xFFF000
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0xD17F_FC62);

        let mut cpu = Cpu::new();
        cpu.write_reg(3, 0x100_0000); // X3 = 0x1000000
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert!(result.is_ok(), "step should succeed");

        // X2 = 0x1000000 - 0xFFF000 = 0x1000
        assert_eq!(cpu.read_reg(2), 0x1000);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_step_movz() {
        // MOVZ X0, #0x42, LSL #0  →  0xD2800840
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0xD280_0840);

        let mut cpu = Cpu::new();
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert!(result.is_ok(), "step should succeed");

        assert_eq!(cpu.read_reg(0), 0x42);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_step_hlt() {
        // HLT #0  →  0xD4400000
        // Memory must be large enough to cover PC (0x100) + 4.
        let mut mem = vec![0u8; 0x104];
        write_inst(&mut mem, 0x100, 0xD440_0000);

        let mut cpu = Cpu::new();
        cpu.write_pc(0x100);

        let result = cpu.step(&mut mem);
        assert_eq!(result, Err(SimError::Halt));
        // PC must not change on HLT.
        assert_eq!(cpu.read_pc(), 0x100);
    }

    #[test]
    fn test_step_multiple() {
        // Sequence:
        //   0x00: MOVZ X5, #0x42  →  0xD2800845  (write 0x42 into X5)
        //   0x04: ADD X5, X5, #1  →  0x910004A5  (X5 = X5 + 1 → 0x43)
        //   0x08: HLT #0          →  0xD4400000

        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0x00, 0xD280_0845); // MOVZ X5, #0x42
        write_inst(&mut mem, 0x04, 0x9100_04A5); // ADD X5, X5, #1
        write_inst(&mut mem, 0x08, 0xD440_0000); // HLT

        let mut cpu = Cpu::new();
        cpu.write_pc(0x00);

        // Step 1: MOVZ
        cpu.step(&mut mem).expect("step 1 (MOVZ)");
        assert_eq!(cpu.read_reg(5), 0x42);
        assert_eq!(cpu.read_pc(), 0x04);

        // Step 2: ADD
        cpu.step(&mut mem).expect("step 2 (ADD)");
        assert_eq!(cpu.read_reg(5), 0x43);
        assert_eq!(cpu.read_pc(), 0x08);

        // Step 3: HLT
        let result = cpu.step(&mut mem);
        assert_eq!(result, Err(SimError::Halt));
        assert_eq!(cpu.read_pc(), 0x08);
    }

    #[test]
    fn test_fetch_out_of_bounds() {
        // Memory has only 4 bytes; PC = 8 is out of bounds.
        let mem = vec![0u8; 4];
        let cpu = Cpu {
            pc: 8,
            ..Cpu::new()
        };
        let result = cpu.fetch(&mem);
        assert!(matches!(result, Err(SimError::MemoryFault(8))));
    }

    #[test]
    fn test_step_unknown_instruction() {
        // 0x00000000 is not a valid decoded instruction → Unknown
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x0000_0000);

        let mut cpu = Cpu::new();
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert_eq!(result, Err(SimError::UnknownInstruction(0x0000_0000)));
        // PC must not advance.
        assert_eq!(cpu.read_pc(), 0);
    }

    #[test]
    fn test_execute_add_sub_rd_31_writes_sp() {
        // ADD SP, X0, #8  →  encoding with rd=31, rn=0, imm12=8
        // sf=1, op=0, S=0, sh=0, imm12=8, rn=0, rd=31
        // = 0x9100201F
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x9100_201F);

        let mut cpu = Cpu::new();
        cpu.write_reg(0, 100);
        cpu.write_sp(0);
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        assert_eq!(cpu.read_sp(), 108); // 100 + 8
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_execute_add_sub_rn_31_reads_sp() {
        // ADD X1, SP, #16  →  rn=31, rd=1, imm12=16
        // sf=1, op=0, sh=0, imm12=16 (0x010), rn=31, rd=1
        // = 0x910043E1
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x9100_43E1);

        let mut cpu = Cpu::new();
        cpu.write_sp(200);
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        assert_eq!(cpu.read_reg(1), 216); // 200 + 16
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_execute_movz_rd_31_writes_sp() {
        // MOVZ SP, #0x42  →  rd=31, hw=0, imm16=0x42
        // Encoding: sf=1 | 10 | 100101 | hw=00 | imm16=0x42 | rd=11111
        // = 0xD280085F
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0xD280_085F);

        let mut cpu = Cpu::new();
        cpu.write_sp(0xFFFF);
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        assert_eq!(cpu.read_sp(), 0x42);
    }

    #[test]
    fn test_execute_add_32bit() {
        // ADD W0, W1, #5  (32-bit) → 0x11001420
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x1100_1420);

        let mut cpu = Cpu::new();
        // Set X1 to a value with upper bits set; 32-bit op should ignore them.
        cpu.write_reg(1, 0xFFFF_FFFF_0000_000A); // lower 32 = 10
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        // Result should be 15, zero-extended to 64 bits.
        assert_eq!(cpu.read_reg(0), 15);
        assert_eq!(cpu.read_pc(), 4);
    }
}
