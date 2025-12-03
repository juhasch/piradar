# Default configuration for BGT60TR13C

default_config = {
    'MAIN': 0x1E8270,  # Address: 0x00
    'ADC0': 0x0A0210,  # Address: 0x01
    'PACR1': 0xE967FD,  # Address: 0x04
    'PACR2': 0x0805B4,  # Address: 0x05
    'PACR2.DIVSET': 20, # Explicitly set DIVSET to default
    'SFCTL': 0x1027FF,  # Address: 0x06
    'SADC_CTRL': 0x010700,  # Address: 0x07
    'CSI_0': 0x000000,  # Address: 0x08
    'CSI_1': 0x000000,  # Address: 0x09
    'CSI_2': 0x000000,  # Address: 0x0A
    'CSCI': 0x000BE0,  # Address: 0x0B
    'CSDS_0': 0x000000,  # Address: 0x0C
    'CSDS_1': 0x000000,  # Address: 0x0D
    'CSDS_2': 0x000000,  # Address: 0x0E
    'CSCDS': 0x000B60,  # Address: 0x0F
    'CSU_0': 0x13FC51,  # Address: 0x10
    'CSU_1': 0x7FF41F,  # Address: 0x11
    'CSU_2': 0x701CE7,  # Address: 0x12
    'CSD_2': 0x000490,  # Address: 0x16
    'CS2': 0x000480,  # Address: 0x1D - Channel set control 2 (up/down)
    'CS3': 0x000480,  # Address: 0x24 - Channel set control 3 (up/down)
    'CS4': 0x000480,  # Address: 0x2B - Channel set control 4 (up/down)
    'CCR0': 0x11BE0E,  # Address: 0x2C
    'CCR1': 0x5D7C0A,  # Address: 0x2D
    'CCR2': 0x03F000,  # Address: 0x2E
    'CCR3': 0x787E1E,  # Address: 0x2F
    'PLL1_0': 0xA89A94,  # Address: 0x30
    'PLL1_1': 0x000262,  # Address: 0x31
    'PLL1_2': 0x0002B2,  # Address: 0x32
    'PLL1_3': 0x000080,  # Address: 0x33
    'PLL1_4': 0x000000,  # Address: 0x34
    'PLL1_5': 0x000000,  # Address: 0x35
    'PLL1_6': 0x000000,  # Address: 0x36
    'PLL1_7': 0x344B10,  # Address: 0x37
    'PLL2_7': 0x000100,  # Address: 0x3F
    'PLL3_7': 0x000100,  # Address: 0x47
    'PLL4_7': 0x000100,  # Address: 0x4F
    'RFT1': 0x000000,  # Address: 0x56 - RF test register 1 (Reserved)
    'SDFT0': 0x000000,  # Address: 0x5B - Design for test register 0
}
