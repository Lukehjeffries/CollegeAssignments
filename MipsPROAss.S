.data
prompt_x: .asciiz "Enter base (x): "
prompt_y: .asciiz "Enter exponent (y): "
result_msg: .asciiz "Result: "
newline: .asciiz "\n"

.text
.globl main

main:
    # Prompt user for x
    li $v0, 4
    la $a0, prompt_x
    syscall

    # Read x
    li $v0, 5
    syscall
    move $t0, $v0   # t0 = x

    # Prompt user for y
    li $v0, 4
    la $a0, prompt_y
    syscall

    # Read y
    li $v0, 5
    syscall
    move $t1, $v0   # t1 = y

    # Initialize result = 1
    li $t2, 1   # t2 = result
    li $t3, 1   # t3 = i (loop counter)

power_loop:
    bgt $t3, $t1, done  # if i > y, exit loop
    mul $t2, $t2, $t0   # result *= x (using pseudo-instruction as per assignment)
    addi $t3, $t3, 1    # i++
    j power_loop

done:
    # Print result message
    li $v0, 4
    la $a0, result_msg
    syscall

    # Print result
    li $v0, 1
    move $a0, $t2
    syscall

    # Print newline
    li $v0, 4
    la $a0, newline
    syscall

    # Exit program
    li $v0, 10
    syscall
