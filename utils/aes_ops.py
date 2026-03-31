"""
AES Operations Module
=====================
Implements AES-128 operations for cryptanalysis experiments:
- S-Box / Inverse S-Box
- SubBytes, ShiftRows, MixColumns, AddRoundKey
- Key Schedule (128-bit)
- Reduced-round AES encryption (configurable 1-10 rounds)
- Full AES-128 encryption

Reference: FIPS 197 (AES Standard)
"""

import numpy as np

# ============================================================
# AES S-Box (Forward)
# ============================================================
SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
], dtype=np.uint8)

# ============================================================
# AES Inverse S-Box
# ============================================================
INV_SBOX = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
], dtype=np.uint8)

# ============================================================
# Round constants for Key Expansion
# ============================================================
RCON = np.array([
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
], dtype=np.uint8)


# ============================================================
# GF(2^8) multiplication helpers for MixColumns
# ============================================================
def _xtime(a):
    """Multiply by x (i.e., by 2) in GF(2^8) with irreducible polynomial x^8+x^4+x^3+x+1."""
    return ((a << 1) ^ (((a >> 7) & 1) * 0x1B)) & 0xFF


def _gf_mult(a, b):
    """Multiply two elements in GF(2^8)."""
    result = 0
    a = int(a)
    b = int(b)
    for _ in range(8):
        if b & 1:
            result ^= a
        a = _xtime(a)
        b >>= 1
    return result & 0xFF


# ============================================================
# Core AES Operations
# ============================================================
def sub_bytes(state):
    """Apply S-Box substitution to each byte in the 4x4 state."""
    return SBOX[state]


def inv_sub_bytes(state):
    """Apply Inverse S-Box substitution."""
    return INV_SBOX[state]


def shift_rows(state):
    """
    Shift rows of the 4x4 state matrix.
    State is a flat array of 16 bytes in column-major order:
    [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]
    
    The 4x4 matrix (column-major):
    s0  s4  s8   s12
    s1  s5  s9   s13
    s2  s6  s10  s14
    s3  s7  s11  s15
    
    Row 0: no shift
    Row 1: shift left 1
    Row 2: shift left 2
    Row 3: shift left 3
    """
    s = state.copy()
    # Row 1: shift left by 1
    s[1], s[5], s[9], s[13] = state[5], state[9], state[13], state[1]
    # Row 2: shift left by 2
    s[2], s[6], s[10], s[14] = state[10], state[14], state[2], state[6]
    # Row 3: shift left by 3
    s[3], s[7], s[11], s[15] = state[15], state[3], state[7], state[11]
    return s


def inv_shift_rows(state):
    """Inverse ShiftRows."""
    s = state.copy()
    # Row 1: shift right by 1
    s[1], s[5], s[9], s[13] = state[13], state[1], state[5], state[9]
    # Row 2: shift right by 2
    s[2], s[6], s[10], s[14] = state[10], state[14], state[2], state[6]
    # Row 3: shift right by 3
    s[3], s[7], s[11], s[15] = state[7], state[11], state[15], state[3]
    return s


def mix_columns(state):
    """
    MixColumns operation on 4x4 state (column-major, flat array of 16 bytes).
    Each column is multiplied by the MixColumns matrix in GF(2^8):
    [2, 3, 1, 1]
    [1, 2, 3, 1]
    [1, 1, 2, 3]
    [3, 1, 1, 2]
    """
    s = state.copy().astype(np.uint8)
    result = np.zeros(16, dtype=np.uint8)
    for col in range(4):
        c = col * 4
        a = s[c:c+4].copy()
        result[c+0] = _gf_mult(2, a[0]) ^ _gf_mult(3, a[1]) ^ a[2] ^ a[3]
        result[c+1] = a[0] ^ _gf_mult(2, a[1]) ^ _gf_mult(3, a[2]) ^ a[3]
        result[c+2] = a[0] ^ a[1] ^ _gf_mult(2, a[2]) ^ _gf_mult(3, a[3])
        result[c+3] = _gf_mult(3, a[0]) ^ a[1] ^ a[2] ^ _gf_mult(2, a[3])
    return result


def inv_mix_columns(state):
    """Inverse MixColumns operation."""
    s = state.copy().astype(np.uint8)
    result = np.zeros(16, dtype=np.uint8)
    for col in range(4):
        c = col * 4
        a = s[c:c+4].copy()
        result[c+0] = _gf_mult(0x0e, a[0]) ^ _gf_mult(0x0b, a[1]) ^ _gf_mult(0x0d, a[2]) ^ _gf_mult(0x09, a[3])
        result[c+1] = _gf_mult(0x09, a[0]) ^ _gf_mult(0x0e, a[1]) ^ _gf_mult(0x0b, a[2]) ^ _gf_mult(0x0d, a[3])
        result[c+2] = _gf_mult(0x0d, a[0]) ^ _gf_mult(0x09, a[1]) ^ _gf_mult(0x0e, a[2]) ^ _gf_mult(0x0b, a[3])
        result[c+3] = _gf_mult(0x0b, a[0]) ^ _gf_mult(0x0d, a[1]) ^ _gf_mult(0x09, a[2]) ^ _gf_mult(0x0e, a[3])
    return result


def add_round_key(state, round_key):
    """XOR state with round key."""
    return (state ^ round_key).astype(np.uint8)


# ============================================================
# Key Expansion (AES-128)
# ============================================================
def key_expansion(key):
    """
    Expand 16-byte key into 11 round keys (for AES-128, 10 rounds + initial).
    
    Args:
        key: np.array of 16 bytes
    
    Returns:
        round_keys: np.array of shape (11, 16)
    """
    key = np.array(key, dtype=np.uint8)
    # Total words needed: 4 * (Nr + 1) = 4 * 11 = 44 words
    nk = 4  # Key length in 32-bit words
    nr = 10  # Number of rounds for AES-128
    
    # Initialize with the original key (4 words)
    w = np.zeros((44, 4), dtype=np.uint8)
    for i in range(nk):
        w[i] = key[i*4:(i+1)*4]
    
    for i in range(nk, 4 * (nr + 1)):
        temp = w[i-1].copy()
        if i % nk == 0:
            # RotWord: rotate left by 1 byte
            temp = np.array([temp[1], temp[2], temp[3], temp[0]], dtype=np.uint8)
            # SubWord: apply S-Box
            temp = SBOX[temp]
            # XOR with round constant
            temp[0] ^= RCON[i // nk - 1]
        w[i] = w[i - nk] ^ temp
    
    # Reshape into 11 round keys of 16 bytes each
    round_keys = np.zeros((nr + 1, 16), dtype=np.uint8)
    for r in range(nr + 1):
        for col in range(4):
            round_keys[r, col*4:(col+1)*4] = w[r*4 + col]
    
    return round_keys


# ============================================================
# AES Encryption (configurable rounds)
# ============================================================
def aes_encrypt(plaintext, key, num_rounds=10):
    """
    AES-128 encryption with configurable number of rounds.
    
    Args:
        plaintext: np.array of 16 bytes
        key: np.array of 16 bytes
        num_rounds: int, number of AES rounds (1-10, default=10 for full AES)
    
    Returns:
        ciphertext: np.array of 16 bytes
    """
    plaintext = np.array(plaintext, dtype=np.uint8)
    key = np.array(key, dtype=np.uint8)
    
    assert len(plaintext) == 16, "Plaintext must be 16 bytes"
    assert len(key) == 16, "Key must be 16 bytes"
    assert 1 <= num_rounds <= 10, "num_rounds must be 1-10"
    
    # Key expansion (generate all 11 round keys)
    round_keys = key_expansion(key)
    
    # Initial round key addition
    state = add_round_key(plaintext, round_keys[0])
    
    # Main rounds
    for r in range(1, num_rounds):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[r])
    
    # Final round (no MixColumns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[num_rounds])
    
    return state


def aes_decrypt(ciphertext, key, num_rounds=10):
    """
    AES-128 decryption with configurable number of rounds.
    
    Args:
        ciphertext: np.array of 16 bytes
        key: np.array of 16 bytes
        num_rounds: int, number of AES rounds (1-10)
    
    Returns:
        plaintext: np.array of 16 bytes
    """
    ciphertext = np.array(ciphertext, dtype=np.uint8)
    key = np.array(key, dtype=np.uint8)
    
    round_keys = key_expansion(key)
    
    # Inverse final round
    state = add_round_key(ciphertext, round_keys[num_rounds])
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    
    # Inverse main rounds
    for r in range(num_rounds - 1, 0, -1):
        state = add_round_key(state, round_keys[r])
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
    
    # Remove initial round key
    state = add_round_key(state, round_keys[0])
    
    return state


# ============================================================
# Batch operations (vectorized for dataset generation)
# ============================================================
def aes_encrypt_batch(plaintexts, key, num_rounds=10):
    """
    Encrypt a batch of plaintexts with the same key.
    
    Args:
        plaintexts: np.array of shape (N, 16)
        key: np.array of 16 bytes
        num_rounds: int
    
    Returns:
        ciphertexts: np.array of shape (N, 16)
    """
    n = len(plaintexts)
    ciphertexts = np.zeros((n, 16), dtype=np.uint8)
    for i in range(n):
        ciphertexts[i] = aes_encrypt(plaintexts[i], key, num_rounds)
    return ciphertexts


def get_sbox_output(plaintext_byte, key_byte):
    """
    Compute AES S-Box output for a given plaintext byte and key byte.
    This is the intermediate value targeted in SCA attacks.
    
    sbox_out = S-Box(plaintext_byte XOR key_byte)
    """
    return SBOX[plaintext_byte ^ key_byte]


def hamming_weight(x):
    """Compute Hamming weight (number of 1-bits) of a byte."""
    x = int(x)
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def hamming_weight_batch(arr):
    """Compute Hamming weight for an array of bytes."""
    hw_table = np.array([hamming_weight(i) for i in range(256)], dtype=np.uint8)
    return hw_table[arr]


# ============================================================
# Verification helper
# ============================================================
def verify_aes():
    """
    Verify AES implementation against known test vectors (FIPS 197).
    """
    # Test vector from FIPS 197 Appendix B
    key = np.array([
        0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6,
        0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C
    ], dtype=np.uint8)
    
    plaintext = np.array([
        0x32, 0x43, 0xF6, 0xA8, 0x88, 0x5A, 0x30, 0x8D,
        0x31, 0x31, 0x98, 0xA2, 0xE0, 0x37, 0x07, 0x34
    ], dtype=np.uint8)
    
    expected_ciphertext = np.array([
        0x39, 0x25, 0x84, 0x1D, 0x02, 0xDC, 0x09, 0xFB,
        0xDC, 0x11, 0x85, 0x97, 0x19, 0x6A, 0x0B, 0x32
    ], dtype=np.uint8)
    
    # Test full AES (10 rounds)
    ciphertext = aes_encrypt(plaintext, key, num_rounds=10)
    assert np.array_equal(ciphertext, expected_ciphertext), \
        f"AES encryption failed!\nExpected: {expected_ciphertext.tolist()}\nGot:      {ciphertext.tolist()}"
    
    # Test decrypt
    decrypted = aes_decrypt(ciphertext, key, num_rounds=10)
    assert np.array_equal(decrypted, plaintext), \
        f"AES decryption failed!\nExpected: {plaintext.tolist()}\nGot:      {decrypted.tolist()}"
    
    # Test reduced-round encrypt/decrypt roundtrip
    for nr in range(1, 11):
        ct = aes_encrypt(plaintext, key, num_rounds=nr)
        pt = aes_decrypt(ct, key, num_rounds=nr)
        assert np.array_equal(pt, plaintext), \
            f"Roundtrip failed for {nr} rounds!"
    
    print("✅ All AES verification tests passed!")
    return True


if __name__ == "__main__":
    verify_aes()
