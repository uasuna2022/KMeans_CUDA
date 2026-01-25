# KMeans CUDA algorithm

## How to run a program?
0. Ensure you have CMake packages installed. On Windows there is x64 Native Tools Command Prompt for VS 2022, where these packages are already installed and ready to be used.
1. Clone the repository to your device.
   ```bash
    git clone https://github.com/uasuna2022/KMeans_CUDA.git <destination_folder>
   ```
2. Enter the folder.
   ```bash
   cd <destination_folder>/KMeansProject/KMeansProject
   ```
3. Create `build` folder and enter it.
   ```bash
   mkdir build
   cd build
   ```
4. Run `cmake` commands.
   ```bash
   cmake ..   # if this command executed with success (build files have been written to <...>), run the next command
   cmake --build . --config Release
   ```
5. If everything finished successfully, go to Release folder.
   ```bash
   cd Release
   ```
6. You may see KMeans.exe file there. Run it with **4** additional parametres:
   - `bin` or `txt` - **the extension** of input file with data
   - `cpu` or `gpu1` or `gpu2` - **computation method** for KMeans algorithm
   - **input** file path
   - **output** file path
   ```bash
   KMeans txt gpu1 ../data/input_file.txt ../data/output_file
   ```
   
   
