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
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
