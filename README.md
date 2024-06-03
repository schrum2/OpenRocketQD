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
For most people jpype will be able to automatically find the JDK. However, if it fails or you want to be sure you are using the correct version, add the JDK path to a JAVA_HOME environment variable:
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

## Usage

The main Python script is `evolve_rockets.py`. At a minimum, it requires a command line
parameter specifying which QD algorithm to apply. Because this code was adapted from 
other examples using Pyribs, the source code mentions some algorithms that are not
fully supported. However, the following commands will work:

+ Plain MAP-Elites
  ```
  python evolve_rockets.py map_elites
  ```
+ Covariance Matrix Adaptation MAP-Elites (What emitters?)
  ```
  python evolve_rockets.py cma_me_imp
  ```
+ CMA-MAE???
  ```
  python evolve_rockets.py cma_mae
  ```

Once evolution completes, results are stored in the subdirectory `evolve_rockets`.
In particular, there will be a `csv` file associated with the algorithm used 
for evolution. This represents the contents of the final archive, and any
rocket in this archive can be evaluated individually using the `rocket_evaluate.py`
script. Here is a example of how to use it:

TODO