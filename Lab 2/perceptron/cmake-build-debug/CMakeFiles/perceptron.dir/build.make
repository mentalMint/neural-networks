# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/mnt/d/Studies/Neuro/Lab 2/perceptron"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/perceptron.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/perceptron.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/perceptron.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/perceptron.dir/flags.make

CMakeFiles/perceptron.dir/main.cpp.o: CMakeFiles/perceptron.dir/flags.make
CMakeFiles/perceptron.dir/main.cpp.o: /mnt/d/Studies/Neuro/Lab\ 2/perceptron/main.cpp
CMakeFiles/perceptron.dir/main.cpp.o: CMakeFiles/perceptron.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/perceptron.dir/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/perceptron.dir/main.cpp.o -MF CMakeFiles/perceptron.dir/main.cpp.o.d -o CMakeFiles/perceptron.dir/main.cpp.o -c "/mnt/d/Studies/Neuro/Lab 2/perceptron/main.cpp"

CMakeFiles/perceptron.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perceptron.dir/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/d/Studies/Neuro/Lab 2/perceptron/main.cpp" > CMakeFiles/perceptron.dir/main.cpp.i

CMakeFiles/perceptron.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perceptron.dir/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/d/Studies/Neuro/Lab 2/perceptron/main.cpp" -o CMakeFiles/perceptron.dir/main.cpp.s

CMakeFiles/perceptron.dir/neuro.cpp.o: CMakeFiles/perceptron.dir/flags.make
CMakeFiles/perceptron.dir/neuro.cpp.o: /mnt/d/Studies/Neuro/Lab\ 2/perceptron/neuro.cpp
CMakeFiles/perceptron.dir/neuro.cpp.o: CMakeFiles/perceptron.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/perceptron.dir/neuro.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/perceptron.dir/neuro.cpp.o -MF CMakeFiles/perceptron.dir/neuro.cpp.o.d -o CMakeFiles/perceptron.dir/neuro.cpp.o -c "/mnt/d/Studies/Neuro/Lab 2/perceptron/neuro.cpp"

CMakeFiles/perceptron.dir/neuro.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/perceptron.dir/neuro.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/d/Studies/Neuro/Lab 2/perceptron/neuro.cpp" > CMakeFiles/perceptron.dir/neuro.cpp.i

CMakeFiles/perceptron.dir/neuro.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/perceptron.dir/neuro.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/d/Studies/Neuro/Lab 2/perceptron/neuro.cpp" -o CMakeFiles/perceptron.dir/neuro.cpp.s

# Object files for target perceptron
perceptron_OBJECTS = \
"CMakeFiles/perceptron.dir/main.cpp.o" \
"CMakeFiles/perceptron.dir/neuro.cpp.o"

# External object files for target perceptron
perceptron_EXTERNAL_OBJECTS =

perceptron: CMakeFiles/perceptron.dir/main.cpp.o
perceptron: CMakeFiles/perceptron.dir/neuro.cpp.o
perceptron: CMakeFiles/perceptron.dir/build.make
perceptron: CMakeFiles/perceptron.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable perceptron"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/perceptron.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/perceptron.dir/build: perceptron
.PHONY : CMakeFiles/perceptron.dir/build

CMakeFiles/perceptron.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/perceptron.dir/cmake_clean.cmake
.PHONY : CMakeFiles/perceptron.dir/clean

CMakeFiles/perceptron.dir/depend:
	cd "/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/mnt/d/Studies/Neuro/Lab 2/perceptron" "/mnt/d/Studies/Neuro/Lab 2/perceptron" "/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug" "/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug" "/mnt/d/Studies/Neuro/Lab 2/perceptron/cmake-build-debug/CMakeFiles/perceptron.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/perceptron.dir/depend
