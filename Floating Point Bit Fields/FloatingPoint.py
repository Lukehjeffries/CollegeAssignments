def ieee754_to_float(sign_bit, exponent_bits, mantissa_bits, precision):
    if precision == "single":
        exponent_length = 8
        mantissa_length = 23
        bias = 127
    elif precision == "double":
        exponent_length = 11
        mantissa_length = 52
        bias = 1023
    else:
        raise ValueError("Unsupported precision")

    sign = int(sign_bit, 2)
    exponent = int(exponent_bits, 2)
    mantissa = int(mantissa_bits, 2)

    if exponent == (1 << exponent_length) - 1:
        if mantissa == 0:
            return float("-inf") if sign else float("inf")
        else:
            return float("nan")

    if exponent == 0:
        exponent_actual = 1 - bias
        is_normalized = False
    else:
        exponent_actual = exponent - bias
        is_normalized = True

    frac = 0.0
    for i in range(len(mantissa_bits)):
        if mantissa_bits[i] == '1':
            frac += 2 ** -(i + 1)

    value = (1 + frac) if is_normalized else frac
    value *= 2 ** exponent_actual

    return -value if sign else value

def main():
    precision = input("Enter precision (single/double): ").strip().lower()

    if precision == "single":
        print("Enter 1-bit sign, 8-bit exponent, and 23-bit mantissa")
        expected_exp = 8
        expected_mantissa = 23
    elif precision == "double":
        print("Enter 1-bit sign, 11-bit exponent, and 52-bit mantissa")
        expected_exp = 11
        expected_mantissa = 52
    else:
        print("Unsupported precision")
        return

    sign = input("Sign (1 bit): ").strip()
    exponent = input(f"Exponent ({expected_exp} bits): ").strip()
    mantissa = input(f"Mantissa ({expected_mantissa} bits): ").strip()

    if len(sign) != 1 or len(exponent) != expected_exp or len(mantissa) != expected_mantissa:
        print("Invalid input lengths.")
        return

    result = ieee754_to_float(sign, exponent, mantissa, precision)
    print("Decimal value:", result)

if __name__ == "__main__":
    main()
