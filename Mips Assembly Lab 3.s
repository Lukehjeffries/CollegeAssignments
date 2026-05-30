.data 
prompt: .asciiz "Enter 1-12 for a number associated with the corresponding month: "

months:     
    .asciiz "January\n\0"
    .asciiz "February\n\0"
    .asciiz "March\n\0"
    .asciiz "April\n\0"
    .asciiz "May\n\0"
    .asciiz "June\n\0"
    .asciiz "July\n\0"
    .asciiz "August\n\0"
    .asciiz "September\n\0"
    .asciiz "October\n\0"
    .asciiz "November\n\0"
    .asciiz "December\n\0"

.text
.globl main

main:
    # Print prompt
    li $v0, 4          
    la $a0, prompt
    syscall

    # Read integer input
    li $v0, 5         
    syscall
    move $t0, $v0     

    # Validate input (1 ≤ input ≤ 12)
    li $t1, 1         
    li $t2, 12        

    blt $t0, $t1, main  # If input < 1, restart
    bgt $t0, $t2, main  # If input > 12, restart

    # Convert input to zero-based index
    addi $t0, $t0, -1

    # Compute offset (multiply index by 10 using shifts)
    sll $t1, $t0, 3    # $t1 = $t0 * 8
    sll $t2, $t0, 1    # $t2 = $t0 * 2
    add $t0, $t1, $t2  # $t0 = $t0 * 10

    # Load base address of months
    la $t4, months
    add $t4, $t4, $t0  # Move to correct month

    # Print the month
    li $v0, 4
    move $a0, $t4
    syscall

    # Exit program
    li $v0, 10
    syscall
