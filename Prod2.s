.data
prompt_a: .asciiz "Enter first number (a): "
prompt_b: .asciiz "Enter second number (b): "
result_msg: .asciiz "GCD: "
newline: .asciiz "\n"

.text
.globl main

main:
    # Prompt for a
    li $v0, 4
    la $a0, prompt_a
    syscall

    # Read a
    li $v0, 5
    syscall
    move $a0, $v0  # Store a in $a0 (first argument)

    # Prompt for b
    li $v0, 4
    la $a0, prompt_b
    syscall

    # Read b
    li $v0, 5
    syscall
    move $a1, $v0  # Store b in $a1 (second argument)

    # Call gcd(a, b)
    jal gcd

    # Print result message
    li $v0, 4
    la $a0, result_msg
    syscall

    # Print result (GCD)
    li $v0, 1
    move $a0, $v0  # Result from gcd is in $v0
    syscall

    # Print newline
    li $v0, 4
    la $a0, newline
    syscall

    # Exit program
    li $v0, 10
    syscall

gcd:
    # Base case: if b == 0, return a
    beqz $a1, gcd_base_case  

    # Save return address ($ra) and $a0 (a) on stack
    addi $sp, $sp, -8
    sw $ra, 4($sp)
    sw $a0, 0($sp)

    # Compute a % b
    div $a0, $a1       # Divide a / b
    mfhi $t0           # Get remainder (a % b)

    # Recursive call: gcd(b, a % b)
    move $a0, $a1      # New a = old b
    move $a1, $t0      # New b = a % b
    jal gcd            # Recursive call

    # Restore registers
    lw $ra, 4($sp)
    lw $a0, 0($sp)
    addi $sp, $sp, 8

    jr $ra  # Return to caller

gcd_base_case:
    move $v0, $a0  # Return a when b == 0
    jr $ra         # Return to caller
