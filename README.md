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

There is also a default `bin_model` setting, which is `stability_altitude`.
So, the above command is equivalent to:
```
python evolve_rockets.py map_elites 5 stability_altitude
```
However, an alternative option is the `stabilitynose_altitude` bin model:
```
python evolve_rockets.py map_elites 5 stabilitynose_altitude
```

Once evolution completes, results are stored in the subdirectory `evolve_rockets_output`.
In particular, there will be a `csv` file associated with the algorithm used 
for evolution and the run number. Basically, all relevant files will have a name with a
prefix like `map_elites_stability_altitude_5` or whatever is appropriate for your chosen algorithm, 
run number, and bin model.

The final `csv` file represents the contents of the final archive, and any
rocket in this archive can be evaluated individually using the `rocket_evaluate.py`
script. Here is a example of how to use it:
```
python rocket_evaluate.py .\evolve_rockets_output\map_elites_stability_altitude_0_archive.csv 2000
```
The final number `2000` in this example refers to a line number from the `csv` file
`map_elites_stability_altitude_0_archive.csv`, which means that the specific rocket defined on that line
will be evaluated. For the chosen rocket, several details about its design will be
printed, and then a plot will be displayed of how its altitude changes over time
in three simulated launches.

If you leave out the line number, then the script will actually show you
a listing of the line numbers for the top 12 "stable" evolved rockets in terms of altitude,
to help you identify interesting line numbers. Note here that "stable" means that any
rocket with a stability score of less than 1.0 is filtered out. 
However, if the bin model is `stabilitynose_altitude` then the output is slightly
different. In this case, there are 2 rockets per nose type, of which there are 6.

Note that there is also a script called `rocket_spread.py` that is used to identify
rockets that achieve various altitudes. Here is an example usage:
```
python rocket_spread.py .\evolve_rockets_output\cma_me_imp_stabilitynose_altitude_2_archive.csv 10 110
```
This command finds rocket line numbers that achieve different altitudes. In this example specifically, 
the user is shown the 6 stable rockets whose altitude is closest to 10 meters without exceeding it, and 
the output continues in increments of 10 meters: 20, 30, 40, and so on up to 110.

Back to `rocket_evaluate.py`: If you want to create an `ork` file representing the rocket so that you can analyze it
further in OpenRocket, then add the name of the file to the end of the command line:
```
python rocket_evaluate.py .\evolve_rockets_output\map_elites_stability_altitude_0_archive.csv 2000 map_elites0_2000.ork
```
This will still perform simulations and display a plot of the altitude over time.
If you just want the `ork` file, you can skip the plot by adding the word `skip`
to the end like this:
```
python rocket_evaluate.py .\evolve_rockets_output\map_elites_stability_altitude_0_archive.csv 2000 map_elites0_2000.ork skip
```

Note that although the produced `ork` files are likely compatible with several versions of OpenRocket,
they are specifically intended to work with version 15.03, which is packaged with this
repository as a `jar` file. Assuming you have configured Java appropriately, you can
launch this version of OpenRocket from within the project directory with the command:
```
java -jar OpenRocket-15.03.jar
```
You can view the source code for OpenRocket version 15.03 [here](https://github.com/openrocket/openrocket/tree/release-15.03)

If you want to quickly create `ork` files for all of the rockets that are returned by
looking at the top altitudes, then you can use the `save_multiple.bat` batch file in Windows.
Here are its expected parameters:
```
save_multiple.bat <csv archive> <output ork file prefix> <line num>*
```
Notice that `<line num>*` has an asterisk indicating that you can list as many space separated line numbers
from the archive file that you like. An example usage looks like this
```
save_multiple.bat evolve_rockets_output\cma_me_imp_stabilitynose_altitude_2_archive.csv cma_me_imp_result 5175 5210 5322
```
This outputs three files named `cma_me_imp_result5175.ork`, `cma_me_imp_result5210.ork`, and `cma_me_imp_result5322.ork`.
