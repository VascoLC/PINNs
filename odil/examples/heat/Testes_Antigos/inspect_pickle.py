# inspect_pickle.py
import pickle
import sys

# Replace this with the actual path to your reference file
# Or pass it as a command-line argument
try:
    file_path = sys.argv[1]
except IndexError:
    print("Usage: python inspect_pickle.py <path_to_your_file.pickle>")
    sys.exit(1)

print(f"Inspecting file: {file_path}")

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("\nFile loaded successfully.")
    print("Type of loaded data:", type(data))

    # Odil saves the state as a dictionary of fields
    if isinstance(data, dict) and 'fields' in data:
        print("\nFound 'fields' dictionary. Available field keys:")
        # The actual array is usually at data['fields'][key]['array']
        for key, field_data in data['fields'].items():
             array_shape = field_data.get('array', None)
             if array_shape is not None:
                 print(f"  - '{key}' with shape: {array_shape.shape}")
             else:
                 print(f"  - '{key}' (but its .array attribute is missing or None)")
    else:
        print("\nCould not find an 'odil.State' structure ('fields' dictionary).")
        print("Top-level keys in the file are:", list(data.keys()))


except Exception as e:
    print(f"\nAn error occurred: {e}")