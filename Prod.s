.data
msg_wps:   .asciiz "Please enter the number of words per sentence:\n"
msg_spw:   .asciiz "Please enter the number of syllables per word:\n"
msg_result: .asciiz "The Flesch-Kincaid Reading Level is: "
newline:   .asciiz "\n"

.text
.globl main

main:
    # Prompt for words per sentence (wps)
    li $v0, 4
    la $a0, msg_wps
    syscall

    # Read integer input for wps
    li $v0, 5
    syscall
    move $a0, $v0   # Store wps in $a0 (argument for procedure)

    # Prompt for syllables per word (spw)
    li $v0, 4
    la $a0, msg_spw
    syscall

    # Read integer input for spw
    li $v0, 5
    syscall
    move $a1, $v0   # Store spw in $a1 (argument for procedure)

    # Call compute_fk procedure
    jal compute_fk
    move $t0, $v0   # Store return value in $t0

    # Print result message
    li $v0, 4
    la $a0, msg_result
    syscall

    # Print the computed integer
    li $v0, 1
    move $a0, $t0
    syscall

    # Print a newline
    li $v0, 4
    la $a0, newline
    syscall

    # Exit program
    li $v0, 10
    syscall

compute_fk:
    # Compute (5 * wps - 12 * spw)
    li $t1, 5
    mul $t2, $t1, $a0  # 5 * wps

    li $t3, 12
    mul $t4, $t3, $a1  # 12 * spw

    sub $v0, $t2, $t4  # (5 * wps) - (12 * spw)

    jr $ra             # Return to caller
