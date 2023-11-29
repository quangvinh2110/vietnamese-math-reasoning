import subprocess

# Function to execute the generated Python code in a subprocess
def execute_code(code):
    # Create a subprocess and execute the code
    process = subprocess.Popen(['python3', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for the process to finish and capture the output
    output, error = process.communicate()
    
    # Decode the output and error messages
    output = output.decode('utf-8')
    error = error.decode('utf-8')
    
    # Return the output and error
    return output, error


# Example usage
generated_code = '''
smallest_double_digit_number = 11
next_number = smallest_double_digit_number + 1
option_A = 8+4
option_B = 7+6
option_C = 9+5
option_D = 16-6
correct_option = None
if option
x
'''

# Execute the generated code in a subprocess
output, error = execute_code(generated_code)

# Print the output and error messages
print("Output:")
print(f"`{output}`")
print("Errors:")
print(error)