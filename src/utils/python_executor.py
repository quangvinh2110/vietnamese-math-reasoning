import re
import sys
import io
import contextlib
import traceback
import time
import signal


INPUT_FUNC_PATTERN = re.compile(r"input\(.*?\)")
# COMMENT_PATTERN = re.compile(r"#.*")
DECIMAL_PATTERN = re.compile(r"\d+\.\d+")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")


def preprocess_code(code: str):
    code = INPUT_FUNC_PATTERN.sub("323", code)
    return code


def normalize_output(output: str) -> str:
    decimal_replace_list = []
    for decimal in DECIMAL_PATTERN.findall(output):
        int_part, frac_part = tuple(decimal.split("."))
        if int(frac_part) == 0:
            new_decimal = int_part
            decimal_replace_list.append(
                (decimal, int_part)
            )
        elif len(frac_part) >= 8:
            new_decimal = round(float(decimal), 8)
            if new_decimal==int(new_decimal):
                decimal_replace_list.append(
                    (decimal, str(int(new_decimal)))
                )
            else:
                decimal_replace_list.append(
                    (decimal, str(new_decimal))
                )
        else: 
            decimal_replace_list.append(
                (decimal, str(float(decimal)))
            )
    for old_decimal, new_decimal in decimal_replace_list:
        output = output.replace(old_decimal, new_decimal)

    return output
    
    
def execute_python_code(code: str, timeout: int = 5):
    code = preprocess_code(code)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    output_capture = io.StringIO()
    succeed = False
    with contextlib.redirect_stdout(output_capture):
        try:
            exec(code, {}, {})
            succeed = True
        except Exception as e: 
            output_capture.write(f"An error occurred {e}\n")
            output_capture.write(traceback.format_exc())
        except TimeoutException as te:
            output_capture.write(str(te))
        finally:
            signal.alarm(0)
    output = output_capture.getvalue().strip()[-512:]
    output = normalize_output(output)
    return output, succeed
