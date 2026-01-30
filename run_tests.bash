#!/bin/bash
#
# Unit Test Runner Script
# Facilita la compilación y ejecución de la batería de pruebas
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CUDA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/cuda" && pwd )"
TEST_DIR="${CUDA_DIR}/test"
TEST_BIN="${TEST_DIR}/test_unit_comprehensive.out"

# Print colored output
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA toolkit not found. Please install CUDA."
        exit 1
    fi
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $NF}')
    print_success "CUDA toolkit found (version $CUDA_VERSION)"
    
    # Check GCC
    if ! command -v g++-12 &> /dev/null; then
        print_error "g++-12 not found. Please install gcc 12."
        exit 1
    fi
    print_success "g++-12 found"
    
    # Check GTest
    if ! pkg-config --exists gtest; then
        print_warning "Google Test not found. Attempting to install..."
        sudo apt-get update
        sudo apt-get install -y libgtest-dev
    fi
    print_success "Google Test available"
    
    # Check OpenCV
    if ! pkg-config --exists opencv4; then
        print_error "OpenCV 4 not found. Please install OpenCV."
        exit 1
    fi
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    print_success "OpenCV found (version $OPENCV_VERSION)"
    
    echo
}

# Compile tests
compile_tests() {
    print_header "Compiling Unit Tests"
    
    cd "${CUDA_DIR}"
    
    print_info "Building with: make -f Makefile.tests"
    
    if make -f Makefile.tests clean 2>&1 | grep -q "error"; then
        print_warning "Clean had some warnings (non-fatal)"
    else
        print_success "Clean completed"
    fi
    
    if make -f Makefile.tests 2>&1; then
        print_success "Compilation successful"
    else
        print_error "Compilation failed"
        exit 1
    fi
    
    if [ ! -f "${TEST_BIN}" ]; then
        print_error "Test binary not found at ${TEST_BIN}"
        exit 1
    fi
    
    print_success "Test binary ready: ${TEST_BIN}"
    echo
}

# List all tests
list_tests() {
    print_header "Available Tests"
    echo
    "${TEST_BIN}" --gtest_list_tests
    echo
}

# Run all tests
run_all_tests() {
    print_header "Running All Tests"
    echo
    "${TEST_BIN}"
    echo
}

# Run tests with verbose output
run_verbose_tests() {
    print_header "Running Tests (Verbose Output)"
    echo
    "${TEST_BIN}" --gtest_print_time=1 --gtest_print_utf8=1 -v
    echo
}

# Run specific test group
run_test_group() {
    local group=$1
    print_header "Running Test Group: $group"
    echo
    "${TEST_BIN}" --gtest_filter="$group*"
    echo
}

# Run with filter
run_filtered_tests() {
    local filter=$1
    print_header "Running Tests Matching Filter: $filter"
    echo
    "${TEST_BIN}" --gtest_filter="$filter"
    echo
}

# Print test statistics
print_statistics() {
    print_header "Test Statistics"
    echo
    TOTAL_TESTS=$(${TEST_BIN} --gtest_list_tests 2>/dev/null | grep -c "  ")
    print_info "Total test cases: $TOTAL_TESTS"
    echo
}

# Print detailed help
print_help() {
    cat << 'EOF'
╔════════════════════════════════════════════════════════╗
║  Unit Test Runner - Image Cipher Test Suite           ║
╚════════════════════════════════════════════════════════╝

USAGE:
    ./run_tests.bash [COMMAND] [OPTIONS]

COMMANDS:
    check               Verify prerequisites
    build               Compile tests only
    run                 Run all tests
    run-verbose         Run tests with verbose output
    run-all             Check prerequisites, build, and run
    list                List all available tests
    stats               Show test statistics
    
    By test group:
        param           Run parameter validation tests
        automata        Run cellular automata tests
        memory          Run memory management tests
        kernel          Run CUDA kernel tests
        encryption      Run encryption tests
        size            Run image size tests
        chaos           Run chaos parameter tests
        edge            Run edge case tests
        integration     Run integration tests
    
    Custom filter (GoogleTest pattern matching):
        filter PATTERN  Run tests matching PATTERN
                        Examples:
                        - "*Encrypt*"        matches all encryption tests
                        - "*Small"           matches tests with "Small"
                        - "*256*"            matches 256x256 tests
                        - "EncryptionTest.*" matches all EncryptionTest tests

OPTIONS:
    --help              Show this help message
    --clean             Clean before building
    --rebuild           Clean and rebuild

EXAMPLES:
    # Run all tests with verbose output
    ./run_tests.bash run-all

    # Run only encryption tests
    ./run_tests.bash encryption

    # Run tests matching a pattern
    ./run_tests.bash filter "*Automata*"

    # Run only edge case tests
    ./run_tests.bash edge

    # List all available tests
    ./run_tests.bash list

    # Check if all prerequisites are installed
    ./run_tests.bash check

EOF
}

# Main script logic
main() {
    local command=${1:-help}
    local option=${2:-}
    
    case $command in
        check)
            check_prerequisites
            print_success "All prerequisites satisfied!"
            ;;
        
        build)
            check_prerequisites
            compile_tests
            print_success "Build complete!"
            ;;
        
        run)
            if [ ! -f "${TEST_BIN}" ]; then
                print_warning "Test binary not found. Building first..."
                compile_tests
            fi
            run_all_tests
            ;;
        
        run-verbose)
            if [ ! -f "${TEST_BIN}" ]; then
                print_warning "Test binary not found. Building first..."
                compile_tests
            fi
            run_verbose_tests
            ;;
        
        run-all)
            check_prerequisites
            compile_tests
            run_all_tests
            print_success "Complete test suite finished!"
            ;;
        
        list)
            if [ ! -f "${TEST_BIN}" ]; then
                print_warning "Test binary not found. Building first..."
                compile_tests
            fi
            list_tests
            ;;
        
        stats)
            if [ ! -f "${TEST_BIN}" ]; then
                print_warning "Test binary not found. Building first..."
                compile_tests
            fi
            print_statistics
            ;;
        
        param)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "ParameterValidationTest"
            ;;
        
        automata)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "CellularAutomataTest"
            ;;
        
        memory)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "MemoryTest"
            ;;
        
        kernel)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "KernelTest"
            ;;
        
        encryption)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "EncryptionTest"
            ;;
        
        size)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "ImageSizeTest"
            ;;
        
        chaos)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "ChaosParameterTest"
            ;;
        
        edge)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "EdgeCaseTest"
            ;;
        
        integration)
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_test_group "IntegrationTest"
            ;;
        
        filter)
            if [ -z "$option" ]; then
                print_error "No filter pattern provided"
                echo "Usage: ./run_tests.bash filter PATTERN"
                exit 1
            fi
            if [ ! -f "${TEST_BIN}" ]; then
                compile_tests
            fi
            run_filtered_tests "$option"
            ;;
        
        --help|-h|help)
            print_help
            ;;
        
        --clean)
            print_header "Cleaning Build Artifacts"
            cd "${CUDA_DIR}"
            make -f Makefile.tests clean
            print_success "Clean completed"
            ;;
        
        --rebuild)
            print_header "Rebuild: Clean + Compile"
            cd "${CUDA_DIR}"
            make -f Makefile.tests clean
            compile_tests
            print_success "Rebuild completed"
            ;;
        
        *)
            print_error "Unknown command: $command"
            echo
            print_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
