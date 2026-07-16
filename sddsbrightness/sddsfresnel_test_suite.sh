#!/bin/bash
#
# Comprehensive Test Suite for sddsfresnel
# Based on existing test patterns and SDDS tools framework
#
# This script tests various aspects of sddsfresnel:
# 1. Small pinhole diffraction (Test1)
# 2. Wide aperture diffraction (Test2) 
# 3. Gaussian source effects (Test4)
# 4. Double-slit interferometry (Test5)
# 5. Polychromatic effects
# 6. Error handling and edge cases
#
# Usage: O.Linux/sddsfresnel_test_suite.sh [test_number]
#        If no test number is provided, all tests are run

# Configuration
PROGRAM_PATH="O.Linux-x86_64/sddsfresnel"
OUTPUT_DIR="/tmp/sddsfresnel_tests"
VERBOSE=1

# Create output directory
mkdir -p $OUTPUT_DIR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")  echo -e "${BLUE}[INFO]${NC} $message" ;;
        "PASS")  echo -e "${GREEN}[PASS]${NC} $message" ;;
        "FAIL")  echo -e "${RED}[FAIL]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
    esac
}

# Function to check if program exists
check_program() {
    if [ ! -f "$PROGRAM_PATH" ]; then
        print_status "FAIL" "Program $PROGRAM_PATH not found"
        exit 1
    fi
    if [ ! -x "$PROGRAM_PATH" ]; then
        print_status "FAIL" "Program $PROGRAM_PATH is not executable"
        exit 1
    fi
    print_status "INFO" "Using program: $PROGRAM_PATH"
}

# Function to test usage display
test_usage() {
    print_status "INFO" "Testing usage display..."
    
    # Test with no arguments
    output=$($PROGRAM_PATH 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ] && echo "$output" | grep -q "sddsfresnel"; then
        print_status "PASS" "Usage display works correctly"
        return 0
    else
        print_status "FAIL" "Usage display failed (exit code: $exit_code)"
        return 1
    fi
}

# Test 1: Small Pinhole Diffraction
test_small_pinhole() {
    print_status "INFO" "Running Test 1: Small Pinhole Diffraction..."
    
    local output_file="$OUTPUT_DIR/test1_small_pinhole.sdds"
    local sourceDistance=22
    local imageDistance=14
    local photonEnergy=12000
    local apertureWidth=0.010
    
    $PROGRAM_PATH "$output_file" \
        -source=distance=${sourceDistance},center=0,width=0 \
        -spectrum=energy=${photonEnergy} \
        -image=distance=${imageDistance},center=0,width=1.2,ngridpoints=1024 \
        -aperture=center=0,width=${apertureWidth} \
        -verbose=1
    sddsplot -col=xImage,ImageIntensity0 $output_file -topline=$output_file &

    # Monochromatic pinhole camera image
    local output_file1="$OUTPUT_DIR/test1_small_pinhole.width"
    local output_file2="$OUTPUT_DIR/test1_small_pinhole.profile"
    $PROGRAM_PATH  $output_file1  -spectrum=energy=12000 \
		-source=distance=22,center=0,width=0.15 -image=distance=14,center=0,width=0.5,ngrid=1024 \
		 -aperture=center=0,width=0.025
    sddssequence -pipe=out -define=xIndex -sequence=begin=-512,number=1025,end=512 | \
           sddsprocess -pipe=in  /tmp/junkFile1.sdds \
	   "-def=col,xSource,0.0012 xIndex * 14 * 22 /,units=mm" \
	   "-def=col,SourceIntensity,xSource 0.15 / sqr -0.5 * exp 2.50663 0.15 * /"

    $PROGRAM_PATH  $output_file2  -spectrum=energy=1200 \
	    -source=distance=22,profile=/tmp/junkFile1.sdds -image=distance=14,center=0,width=0.5,ngrid=1024   \
	     -aperture=center=0,width=0.025

    sddsplot  -col=xImage,ImageIntensity $output_file1 $output_file2 -grap=line,vary -leg=file,edit=43d  &

     #ploychromatic pinhole camera
     local output_file3="$OUTPUT_DIR/test1_small_pinhole.mono"
     local output_file4="$OUTPUT_DIR/test1_small_pinhole.bandwidth"
     echo $output_file3
     $PROGRAM_PATH  $output_file3 -spectrum=wavelength=0.1,bandwidth=0.2  \
		    -source=distance=10,center=0,width=0.01 -image=distance=10,center=0,width=0.4,ngrid=1024 \
		    -aperture=center=0,width=0.025
    
     
     sddssequence -pipe=out -def=wIndex -seq=begin=-3.5,number=25,end=3.5 | \
	 sddsprocess  -pipe=in /tmp/junkFile1a.sdds \
		      "-def=col,Wavelength,wIndex 0.02 * 0.1 +,units=nm" \
		      "-def=col,Weight,Wavelength 0.1 - 0.02 / sqr -0.5 * exp"
     
     $PROGRAM_PATH  $output_file4  -spectrum=profile=/tmp/junkFile1a.sdds \
		    -source=distance=10,center=0,width=0.01 -image=distance=10,center=0,width=0.4,ngrid=1024 \
		    -aperture=center=0,width=0.025
     sddsplot -col=xImage,ImageIntensity  -grap=line,vary $output_file3 $output_file4  -leg=file,edit=43d  &
     

    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        print_status "PASS" "Small pinhole diffraction test completed"
        
        # Verify output file has expected columns
        if command -v sddsquery >/dev/null 2>&1; then
            sddsquery "$output_file" > "$OUTPUT_DIR/test1_info.txt" 2>&1
            if grep -q "xImage\|ImageIntensity" "$OUTPUT_DIR/test1_info.txt"; then
                print_status "PASS" "Output file contains expected columns"
            else
                print_status "WARN" "Output file may be missing expected columns"
            fi
        fi
        return 0
    else
        print_status "FAIL" "Small pinhole diffraction test failed"
        return 1
    fi
    
}

# Test 2: Wide Aperture Diffraction
test_wide_aperture() {
    print_status "INFO" "Running Test 2: Wide Aperture Diffraction..."
    
    local output_file="$OUTPUT_DIR/test2_wide_aperture.sdds"
    local sourceDistance=10
    local imageDistance=10
    local photonEnergy=12000
    local apertureWidth=0.100
    
    $PROGRAM_PATH "$output_file" \
        -source=distance=${sourceDistance},center=0,width=0 \
        -spectrum=energy=${photonEnergy} \
        -image=distance=${imageDistance},center=0,width=0.5,ngridpoints=1024 \
        -aperture=center=0,width=${apertureWidth} \
        -verbose=1
    sddsplot -col=xImage,ImageIntensity -graph=line $output_file

    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        print_status "PASS" "Wide aperture diffraction test completed"
        return 0
    else
        print_status "FAIL" "Wide aperture diffraction test failed"
        return 1
    fi
}

# Test 3: Gaussian Source Effects
test_gaussian_source() {
    print_status "INFO" "Running Test 3: Gaussian Source Effects..."
    
    local output_file="$OUTPUT_DIR/test3_gaussian_source.sdds"
    local sourceWidth=0.15
    local sourceDistance=22
    local imageDistance=14
    local photonEnergy=12000
    local apertureWidth=0.025
    
    $PROGRAM_PATH "$output_file" \
        -source=distance=${sourceDistance},center=0,width=${sourceWidth} \
        -spectrum=energy=${photonEnergy} \
        -image=distance=${imageDistance},center=0,width=1.2,ngridpoints=1024 \
        -aperture=center=0,width=${apertureWidth} \
        -verbose=1
     sddsplot -col=xImage,ImageIntensity -graph=line $output_file
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        print_status "PASS" "Gaussian source effects test completed"
        return 0
    else
        print_status "FAIL" "Gaussian source effects test failed"
        return 1
    fi
}

# Test 4: Double-Slit Interferometry
test_double_slit() {
    print_status "INFO" "Running Test 4: Double-Slit Interferometry..."
    
    local output_file="$OUTPUT_DIR/test4_double_slit.sdds"
    local sourceWidth=0.01
    local sourceDistance=22
    local imageDistance=14
    local photonEnergy=12000
    local apertureWidth=0.004
    
    $PROGRAM_PATH "$output_file" \
        -source=distance=${sourceDistance},center=0,width=${sourceWidth} \
        -spectrum=energy=${photonEnergy} \
        -image=distance=${imageDistance},center=0,width=1.2,ngridpoints=1024 \
        -aperture=center=0.025,width=${apertureWidth} \
        -aperture=center=-0.025,width=${apertureWidth} \
        -verbose=1
     sddsplot -col=xImage,ImageIntensity -graph=line $output_file
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        print_status "PASS" "Double-slit interferometry test completed"
        return 0
    else
        print_status "FAIL" "Double-slit interferometry test failed"
        return 1
    fi
}

# Test 5: Polychromatic Effects
test_polychromatic() {
    print_status "INFO" "Running Test 5: Polychromatic Effects..."
    
    local output_file="$OUTPUT_DIR/test5_polychromatic.sdds"
    local sourceDistance=10
    local imageDistance=10
    local photonEnergy=12000
    local bandwidth=0.01
    local nwaves=25
    
    $PROGRAM_PATH "$output_file" \
        -source=distance=${sourceDistance},center=0,width=0 \
        -spectrum=energy=${photonEnergy},bandwidth=${bandwidth},nwaves=${nwaves} \
        -image=distance=${imageDistance},center=0,width=0.5,ngridpoints=512 \
        -aperture=center=0,width=0.050 \
        -verbose=1
    sddsplot -col=xImage,ImageIntensity -graph=line $output_file
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        print_status "PASS" "Polychromatic effects test completed"
        return 0
    else
        print_status "FAIL" "Polychromatic effects test failed"
        return 1
    fi
}

# Test 6: Wavelength vs Energy Specification
test_wavelength_energy() {
    print_status "INFO" "Running Test 6: Wavelength vs Energy Specification..."
    
    # Test with wavelength
    local output_file1="$OUTPUT_DIR/test6a_wavelength.sdds"
    $PROGRAM_PATH "$output_file1" \
        -source=distance=10,center=0,width=0 \
        -spectrum=wavelength=0.1 \
        -image=distance=10,center=0,width=0.5,ngridpoints=256 \
        -aperture=center=0,width=0.050 \
        -verbose=1
    sddsplot -col=xImage,ImageIntensity -graph=line $output_file1
    
    # Test with energy (should give same wavelength: 1239.8418/12398.418 ≈ 0.1 nm)
    local output_file2="$OUTPUT_DIR/test6b_energy.sdds"
    $PROGRAM_PATH "$output_file2" \
        -source=distance=10,center=0,width=0 \
        -spectrum=energy=12398.418 \
        -image=distance=10,center=0,width=0.5,ngridpoints=256 \
        -aperture=center=0,width=0.050 \
        -verbose=1
    sddsplot -col=xImage,ImageIntensity -graph=line $output_file2
    
    if [ $? -eq 0 ] && [ -f "$output_file1" ] && [ -f "$output_file2" ]; then
        print_status "PASS" "Wavelength vs energy specification test completed"
        return 0
    else
        print_status "FAIL" "Wavelength vs energy specification test failed"
        return 1
    fi
}

# Test 7: Error Handling
test_error_handling() {
    print_status "INFO" "Running Test 7: Error Handling..."
    
    local test_passed=0
    
    # Test with invalid parameters
    print_status "INFO" "  Testing invalid energy value..."
    $PROGRAM_PATH "$OUTPUT_DIR/test7_error.sdds" \
        -spectrum=energy=-1000 \
        -image=distance=10,center=0,width=0.5,ngridpoints=256 \
        -aperture=center=0,width=0.050 \
        >/dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        print_status "PASS" "  Correctly rejected negative energy"
        ((test_passed++))
    else
        print_status "FAIL" "  Failed to reject negative energy"
    fi
    
    # Test with missing output file
    print_status "INFO" "  Testing missing output file..."
    $PROGRAM_PATH \
        -spectrum=energy=12000 \
        -image=distance=10,center=0,width=0.5,ngridpoints=256 \
        -aperture=center=0,width=0.050 \
        >/dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        print_status "PASS" "  Correctly rejected missing output file"
        ((test_passed++))
    else
        print_status "FAIL" "  Failed to reject missing output file"
    fi
    
    if [ $test_passed -eq 2 ]; then
        print_status "PASS" "Error handling test completed"
        return 0
    else
        print_status "FAIL" "Error handling test failed ($test_passed/2 passed)"
        return 1
    fi
}

# Test 8: Symmetry Option
test_symmetry() {
    print_status "INFO" "Running Test 8: Symmetry Option..."
    
    local output_file="$OUTPUT_DIR/test8_symmetry.sdds"
    
    $PROGRAM_PATH "$output_file" \
        -source=distance=10,center=0,width=0 \
        -spectrum=energy=12000 \
        -image=distance=10,center=0,width=1.0,ngridpoints=512 \
        -aperture=center=0.1,width=0.020,symmetry=1 \
        -verbose=1
    
    sddsplot -col=xImage,ImageIntensity -graph=line $output_file
    
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        print_status "PASS" "Symmetry option test completed"
        return 0
    else
        print_status "FAIL" "Symmetry option test failed"
        return 1
    fi
}

# Main test runner
run_tests() {
    local test_number=$1
    local total_tests=0
    local passed_tests=0
    
    print_status "INFO" "Starting sddsfresnel test suite..."
    print_status "INFO" "Output directory: $OUTPUT_DIR"
    
    # Check if program exists
    check_program
    
    # Define test functions and names
    declare -a test_functions=("test_usage" "test_small_pinhole" "test_wide_aperture" 
                              "test_gaussian_source" "test_double_slit" "test_polychromatic"
                              "test_wavelength_energy" "test_error_handling" "test_symmetry")
    declare -a test_names=("Usage Display" "Small Pinhole" "Wide Aperture" 
                          "Gaussian Source" "Double-Slit" "Polychromatic"
                          "Wavelength/Energy" "Error Handling" "Symmetry")
    
    # Run specific test or all tests
    if [ -n "$test_number" ] && [ "$test_number" -ge 1 ] && [ "$test_number" -le ${#test_functions[@]} ]; then
        print_status "INFO" "Running single test: ${test_names[$((test_number-1))]}"
        ${test_functions[$((test_number-1))]}
        if [ $? -eq 0 ]; then
            print_status "PASS" "Test $test_number completed successfully"
        else
            print_status "FAIL" "Test $test_number failed"
        fi
    else
        # Run all tests
        for i in "${!test_functions[@]}"; do
            ((total_tests++))
            echo
            print_status "INFO" "Running Test $((i+1)): ${test_names[i]}"
            ${test_functions[i]}
            if [ $? -eq 0 ]; then
                ((passed_tests++))
            fi
        done
        
        echo
        print_status "INFO" "Test Summary: $passed_tests/$total_tests tests passed"
        
        if [ $passed_tests -eq $total_tests ]; then
            print_status "PASS" "All tests completed successfully!"
            exit 0
        else
            print_status "FAIL" "Some tests failed. Check output above for details."
            exit 1
        fi
    fi
}

# Print help
print_help() {
    echo "sddsfresnel Test Suite"
    echo
    echo "Usage: $0 [test_number]"
    echo
    echo "Available tests:"
    echo "  1. Usage Display"
    echo "  2. Small Pinhole Diffraction"
    echo "  3. Wide Aperture Diffraction"
    echo "  4. Gaussian Source Effects"
    echo "  5. Double-Slit Interferometry"
    echo "  6. Polychromatic Effects"
    echo "  7. Wavelength vs Energy Specification"
    echo "  8. Error Handling"
    echo "  9. Symmetry Option"
    echo
    echo "If no test number is provided, all tests will be run."
    echo
    echo "Environment variables:"
    echo "  PROGRAM_PATH - Path to sddsfresnel executable (default: ./sddsfresnel)"
    echo "  OUTPUT_DIR   - Directory for test outputs (default: /tmp/sddsfresnel_tests)"
    echo
}

# Main script logic
case "$1" in
    -h|--help)
        print_help
        exit 0
        ;;
    "")
        run_tests
        ;;
    [1-9])
        run_tests "$1"
        ;;
    *)
        print_status "FAIL" "Invalid test number: $1"
        print_help
        exit 1
        ;;
esac
