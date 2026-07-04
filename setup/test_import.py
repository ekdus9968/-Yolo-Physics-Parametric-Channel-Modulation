try:
    import cv2
    print("✓ cv2 imported")

    import numpy
    print("✓ numpy imported")

    import pandas
    print("✓ pandas imported")

    import matplotlib
    print("✓ matplotlib imported")

    import scipy
    print("✓ scipy imported")

    print("\n🎉 All packages imported successfully!")

except Exception as e:
    import traceback
    print("\n❌ Import failed:")
    traceback.print_exc()
