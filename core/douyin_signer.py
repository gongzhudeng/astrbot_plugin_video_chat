from __future__ import annotations

import hashlib
import random
import time

# Adapted from Johnserf-Seed/f2's Apache-2.0 licensed ABogus implementation.
# This trimmed version only implements GET signing used by Douyin comment APIs.

_BASE64_ALPHABET = "Dkdpgh2ZmsQB80/MfvV36XI1R45-WUAlEixNLwoqYTOPuzKFjJnry79HbGcaStCe"
_UA_BASE64_ALPHABET = "ckdp1h4ZKsUB80/Mfvw36XIgR25+WQAlEi7NLboqYTOPuzmFjJnryx9HVGDaStCe"
_BIG_ARRAY = [
    121,
    243,
    55,
    234,
    103,
    36,
    47,
    228,
    30,
    231,
    106,
    6,
    115,
    95,
    78,
    101,
    250,
    207,
    198,
    50,
    139,
    227,
    220,
    105,
    97,
    143,
    34,
    28,
    194,
    215,
    18,
    100,
    159,
    160,
    43,
    8,
    169,
    217,
    180,
    120,
    247,
    45,
    90,
    11,
    27,
    197,
    46,
    3,
    84,
    72,
    5,
    68,
    62,
    56,
    221,
    75,
    144,
    79,
    73,
    161,
    178,
    81,
    64,
    187,
    134,
    117,
    186,
    118,
    16,
    241,
    130,
    71,
    89,
    147,
    122,
    129,
    65,
    40,
    88,
    150,
    110,
    219,
    199,
    255,
    181,
    254,
    48,
    4,
    195,
    248,
    208,
    32,
    116,
    167,
    69,
    201,
    17,
    124,
    125,
    104,
    96,
    83,
    80,
    127,
    236,
    108,
    154,
    126,
    204,
    15,
    20,
    135,
    112,
    158,
    13,
    1,
    188,
    164,
    210,
    237,
    222,
    98,
    212,
    77,
    253,
    42,
    170,
    202,
    26,
    22,
    29,
    182,
    251,
    10,
    173,
    152,
    58,
    138,
    54,
    141,
    185,
    33,
    157,
    31,
    252,
    132,
    233,
    235,
    102,
    196,
    191,
    223,
    240,
    148,
    39,
    123,
    92,
    82,
    128,
    109,
    57,
    24,
    38,
    113,
    209,
    245,
    2,
    119,
    153,
    229,
    189,
    214,
    230,
    174,
    232,
    63,
    52,
    205,
    86,
    140,
    66,
    175,
    111,
    171,
    246,
    133,
    238,
    193,
    99,
    60,
    74,
    91,
    225,
    51,
    76,
    37,
    145,
    211,
    166,
    151,
    213,
    206,
    0,
    200,
    244,
    176,
    218,
    44,
    184,
    172,
    49,
    216,
    93,
    168,
    53,
    21,
    183,
    41,
    67,
    85,
    224,
    155,
    226,
    242,
    87,
    177,
    146,
    70,
    190,
    12,
    162,
    19,
    137,
    114,
    25,
    165,
    163,
    192,
    23,
    59,
    9,
    94,
    179,
    107,
    35,
    7,
    142,
    131,
    239,
    203,
    149,
    136,
    61,
    249,
    14,
    156,
]
_SORT_INDEX = [
    18,
    20,
    52,
    26,
    30,
    34,
    58,
    38,
    40,
    53,
    42,
    21,
    27,
    54,
    55,
    31,
    35,
    57,
    39,
    41,
    43,
    22,
    28,
    32,
    60,
    36,
    23,
    29,
    33,
    37,
    44,
    45,
    59,
    46,
    47,
    48,
    49,
    50,
    24,
    25,
    65,
    66,
    70,
    71,
]
_SORT_INDEX_XOR = [
    18,
    20,
    26,
    30,
    34,
    38,
    40,
    42,
    21,
    27,
    31,
    35,
    39,
    41,
    43,
    22,
    28,
    32,
    36,
    23,
    29,
    33,
    37,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    24,
    25,
    52,
    53,
    54,
    55,
    57,
    58,
    59,
    60,
    65,
    66,
    70,
    71,
]


def _sm3_bytes(value: str | bytes) -> list[int]:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return list(hashlib.new("sm3", raw).digest())


def _double_sm3(value: str) -> list[int]:
    return _sm3_bytes(bytes(_sm3_bytes(value + "cus")))


def _rc4(key: bytes, plaintext: bytes) -> bytes:
    state = list(range(256))
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % len(key)]) % 256
        state[i], state[j] = state[j], state[i]
    i = j = 0
    output = bytearray()
    for value in plaintext:
        i = (i + 1) % 256
        j = (j + state[i]) % 256
        state[i], state[j] = state[j], state[i]
        output.append(value ^ state[(state[i] + state[j]) % 256])
    return bytes(output)


def _base64_with_alphabet(raw: bytes, alphabet: str) -> str:
    bits = "".join(f"{value:08b}" for value in raw)
    padding = (6 - len(bits) % 6) % 6
    bits += "0" * padding
    encoded = "".join(alphabet[int(bits[i : i + 6], 2)] for i in range(0, len(bits), 6))
    return encoded + "=" * (padding // 2)


def _custom_base64(raw: bytes) -> str:
    result: list[str] = []
    for index in range(0, len(raw), 3):
        chunk = raw[index : index + 3]
        number = int.from_bytes(chunk.ljust(3, b"\0"), "big")
        result.append(_BASE64_ALPHABET[(number >> 18) & 63])
        result.append(_BASE64_ALPHABET[(number >> 12) & 63])
        if len(chunk) > 1:
            result.append(_BASE64_ALPHABET[(number >> 6) & 63])
        if len(chunk) > 2:
            result.append(_BASE64_ALPHABET[number & 63])
    result.append("=" * ((4 - len(result) % 4) % 4))
    return "".join(result)


def _random_prefix() -> bytes:
    output = bytearray()
    for _ in range(3):
        value = int(random.random() * 10000)
        output.extend(
            (
                ((value & 255) & 170) | 1,
                ((value & 255) & 85) | 2,
                ((value >> 8) & 170) | 5,
                ((value >> 8) & 85) | 40,
            )
        )
    return bytes(output)


def _browser_fingerprint() -> str:
    inner_width = random.randint(1024, 1920)
    inner_height = random.randint(768, 1080)
    outer_width = inner_width + random.randint(24, 32)
    outer_height = inner_height + random.randint(75, 90)
    screen_y = random.choice((0, 30))
    size_width = random.randint(1024, 1920)
    size_height = random.randint(768, 1080)
    available_width = random.randint(1280, 1920)
    available_height = random.randint(800, 1080)
    return "|".join(
        str(value)
        for value in (
            inner_width,
            inner_height,
            outer_width,
            outer_height,
            0,
            screen_y,
            0,
            0,
            size_width,
            size_height,
            available_width,
            available_height,
            inner_width,
            inner_height,
            24,
            24,
            "Win32",
        )
    )


def _transform(values: list[int]) -> bytes:
    table = list(_BIG_ARRAY)
    index_b = table[1]
    initial = 0
    previous = 0
    result = bytearray()
    for index, char_value in enumerate(values):
        if index == 0:
            initial = table[index_b]
            total = index_b + initial
            table[1] = initial
            table[index_b] = index_b
        else:
            total = initial + previous
        total %= len(table)
        result.append(char_value ^ table[total])
        previous = table[(index + 2) % len(table)]
        total = (index_b + previous) % len(table)
        initial = table[total]
        table[total] = table[(index + 2) % len(table)]
        table[(index + 2) % len(table)] = initial
        index_b = total
    return bytes(result)


def generate_a_bogus(params: str, user_agent: str) -> str:
    fingerprint = _browser_fingerprint()
    start = int(time.time() * 1000)
    params_hash = _double_sm3(params)
    body_hash = _double_sm3("")
    ua_cipher = _rc4(b"\x00\x01\x0e", user_agent.encode("utf-8"))
    ua_hash = _sm3_bytes(_base64_with_alphabet(ua_cipher, _UA_BASE64_ALPHABET))
    end = int(time.time() * 1000)

    values: dict[int, int] = {
        8: 3,
        18: 44,
        19: 1,
        20: (start >> 24) & 255,
        21: (start >> 16) & 255,
        22: (start >> 8) & 255,
        23: start & 255,
        24: (start >> 32) & 255,
        25: (start >> 40) & 255,
        26: 0,
        27: 0,
        28: 0,
        29: 0,
        30: 0,
        31: 1,
        32: 0,
        33: 0,
        34: 0,
        35: 0,
        36: 0,
        37: 14,
        38: params_hash[21],
        39: params_hash[22],
        40: body_hash[21],
        41: body_hash[22],
        42: ua_hash[23],
        43: ua_hash[24],
        44: (end >> 24) & 255,
        45: (end >> 16) & 255,
        46: (end >> 8) & 255,
        47: end & 255,
        48: 3,
        49: (end >> 32) & 255,
        50: (end >> 40) & 255,
        52: 0,
        53: 0,
        54: 0,
        55: 0,
        57: 6383 & 255,
        58: (6383 >> 8) & 255,
        59: (6383 >> 16) & 255,
        60: (6383 >> 24) & 255,
        65: len(fingerprint),
        66: 0,
        70: 0,
        71: 0,
    }
    sorted_values = [values.get(index, 0) for index in _SORT_INDEX]
    sorted_values.extend(fingerprint.encode("utf-8"))
    checksum = 0
    for index in _SORT_INDEX_XOR:
        checksum ^= values.get(index, 0)
    sorted_values.append(checksum)
    return _custom_base64(_random_prefix() + _transform(sorted_values))
