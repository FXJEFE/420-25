import os
import chardet

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    except Exception as e:
        print(f'Error detecting encoding for {file_path}: {str(e)}')
        return None

def convert_to_utf8(input_path, output_path):
    # Detect encoding
    detected_encoding = detect_encoding(input_path)
    if detected_encoding is None:
        print(f'Error: Could not determine encoding for {input_path}')
        return False
    
    # Try decoding with detected encoding or fallbacks
    encodings = [detected_encoding, 'utf-8', 'latin1', 'ascii']
    content = None
    used_encoding = None
    
    for enc in encodings:
        if enc is None:
            continue
        try:
            with open(input_path, 'r', encoding=enc) as infile:
                content = infile.read()
                used_encoding = enc
            print(f'Successfully decoded {input_path} with {used_encoding}')
            break
        except Exception as e:
            print(f'Tried encoding {enc} for {input_path}: {str(e)}')
    
    # Fallback: Read raw bytes and ignore errors
    if content is None:
        try:
            with open(input_path, 'rb') as infile:
                raw_data = infile.read()
            content = raw_data.decode('utf-8', errors='ignore')
            used_encoding = 'utf-8 (with ignored errors)'
            print(f'Fallback: Decoded {input_path} with UTF-8, ignoring errors')
        except Exception as e:
            print(f'Error: Could not decode {input_path} even with fallback: {str(e)}')
            return False
    
    # Write to output file in UTF-8
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
        print(f'Successfully converted {input_path} to UTF-8 (from {used_encoding})')
        return True
    except Exception as e:
        print(f'Error writing {output_path}: {str(e)}')
        return False

# List of files to convert
files_to_convert = [
    'log.txt',
    'FXJEFE_Features.csv',
    'FXJEFE_trades_outcomes.csv',
    'FXJEFE_trades.csv'
]

# Base directory for MT5 files
base_dir = 'C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Files'

# Convert each file
for file in files_to_convert:
    input_path = os.path.join(base_dir, file)
    if os.path.exists(input_path):
        convert_to_utf8(input_path, input_path)
    else:
        print(f'File not found: {input_path}')