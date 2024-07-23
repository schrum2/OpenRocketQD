# Quality Diversity in Open Rocket

This code evolves model rocket designs using Pyribs and evaluates them using
the OpenRocket simulator.

## Prerequisites
- Java JDK 1.8 (note that an older version of Java is required)
     - [Open JDK 1.8](https://github.com/ojdkbuild/ojdkbuild)
     - [Oracle JDK 8](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html) (requires signup)
     - Ubuntu: `sudo apt-get install openjdk-8-jre`
- Python >= 3.10

## Installation

Install necessary Python packages with:
```
pip install -r requirements.txt
```
You will also need to assure that the environment variable 
`JAVA_HOME` is set to the path for your JDK 1.8 installation.

## Setup JDK

### Linux
For most people jpype will be able to automatically find the JDK. However, if it fails or you want to be sure you are using the correct version, add the JDK path to a `JAVA_HOME` environment variable:
- Find installation directory (e.g. `/usr/lib/jvm/[YOUR JDK 1.8 FOLDER HERE]`)
- Open the `~/.bashrc` file with your favorite text editor (will likely need sudo privileges)
- Add the following line to the `~/.bashrc` file:
    ```
    export JAVA_HOME="/usr/lib/jvm/[YOUR JDK 1.8 FOLDER HERE]"
    ```
- Restart your terminal or run the following for the changes to take effect:
    ```
    source ~/.bashrc
    ```

### Windows

- Set Windows environment variables to the following:
    - Oracle
        ```
        JAVA_HOME = C:\Program Files\Java\[YOUR JDK 1.8 FOLDER HERE]
        ```
    - OpenJDK
        ```
        JAVA_HOME = C:\Program Files\ojdkbuild\[YOUR JDK 1.8 FOLDER HERE]
        ```
- These variables can be set in Windows Advanced System Settings, or from the Powershell with the command:
   ```
   $env:JAVA_HOME = "C:\Program Files\Java\[JDK folder]"
   ```

## Usage

The main Python script is `evolve_rockets.py`. At a minimum, it requires a command line
parameter specifying which QD algorithm to apply. Because this code was adapted from 
other examples using Pyribs, the source code mentions some algorithms that are not
fully supported. However, the following commands will work:

+ Plain MAP-Elites
  ```
  python evolve_rockets.py map_elites
  ```
+ Covariance Matrix Adaptation MAP-Elites (TODO: More details)
  ```
  python evolve_rockets.py cma_me_imp
  ```
+ CMA-MAE (TODO: More details)
  ```
  python evolve_rockets.py cma_mae
  ```

You can optionally have a run number after the algorithm name, such as:
```
python evolve_rockets.py map_elites 5
```
If there is no run number, then it defaults to a value of 0.
 
Once evolution completes, results are stored in the subdirectory `evolve_rockets_output`.
In particular, there will be a `csv` file associated with the algorithm used 
for evolution and the run number. Basically, all relevant files will have a name with a
prefix like `map_elites5` or whatever is appropriate for your chosen algorithm and run number.

The final `csv` file represents the contents of the final archive, and any
rocket in this archive can be evaluated individually using the `rocket_evaluate.py`
script. Here is a example of how to use it:
```
python rocket_evaluate.py .\evolve_rockets_output\map_elites0_archive.csv 2000
```
The final number `2000` in this example refers to a line number from the `csv` file
`map_elites0_archive.csv`, which means that the specific rocket defined on that line
will be evaluated. For the chosen rocket, several details about its design will be
printed, and then a plot will be displayed of how its altitude changes over time
in three simulated launches.

If you leave out the line number, then the script will actually show you
a listing of the line numbers for the top 10 evolved rockets in terms of altitude,
to help you identify interesting line numbers.

If you want to create an `ork` file representing the rocket so that you can analyze it
further in OpenRocket, then add the name of the file to the end of the command line:
```
python rocket_evaluate.py .\evolve_rockets_output\map_elites0_archive.csv 2000 map_elites0_2000.ork
```
This will still perform simulations and display a plot of the altitude over time.
If you just want the `ork` file, you can skip the plot by adding the word `skip`
to the end like this:
```
python rocket_evaluate.py .\evolve_rockets_output\map_elites0_archive.csv 2000 map_elites0_2000.ork skip
```

Note that although the produced `ork` files are likely compatible with several versions of OpenRocket,
they are specifically intended to work with version 15.03, which is packaged with this
repository as a `jar` file. Assuming you have configured Java appropriately, you can
launch this version of OpenRocket from within the project directory with the command:
```
java -jar OpenRocket-15.03.jar
```
