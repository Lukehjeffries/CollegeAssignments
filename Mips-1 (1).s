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
    move $t0, $v0   # Store wps in $t0
    
    # Prompt for syllables per word (spw)
    li $v0, 4
    la $a0, msg_spw
    syscall
    
    # Read integer input for spw
    li $v0, 5
    syscall
    move $t1, $v0   # Store spw in $t1
    
    # Compute the formula: (5 * wps - 12 * spw)
    li $t2, 5        # Load constant 5
    mul $t3, $t2, $t0 # t3 = 5 * wps
    
    li $t4, 12       # Load constant 12
    mul $t5, $t4, $t1 # t5 = 12 * spw
    
    sub $t6, $t3, $t5 # t6 = (5 * wps) - (12 * spw)
    
    # Print result message
    li $v0, 4
    la $a0, msg_result
    syscall
    
    # Print the computed integer
    li $v0, 1
    move $a0, $t6
    syscall
    
    # Print a newline
    li $v0, 4
    la $a0, newline
    syscall
    
    # Exit program
    li $v0, 10
    syscall
