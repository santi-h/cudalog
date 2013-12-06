TARGET = cudalog.lib
OBJECTS = 	devicelog.obj \
            hostlog.obj

.SUFFIXES : .cu .obj 
IGNOREW = -wd4514,-wd4505,-wd4820,-wd4365,-wd4986,-wd4710,-wd4668,-wd4555,-wd4100
OPTIONS = -Xcompiler=-Wall,$(IGNOREW) -arch=sm_20

# LINK OBJECTS
$(TARGET): $(OBJECTS)
    @echo linking $@...
	@nvcc $(OPTIONS) -lib -o=$@ $**

# CREATE OBJECTS
.cu.obj:
    @echo compiling $@...
	@nvcc $(OPTIONS) -dc -o=$@ "$*.cu"

# RUN
run: $(TARGET)
	@cls
	@.\$(TARGET)

# RELEASE
release: $(TARGET)
    @copy $(TARGET) lib\$(TARGET)
    @copy hostlog.h include\hostlog.h
    @copy devicelog.h include\devicelog.h

# CLEAN
clean: 
	@del .\$(TARGET)
	@del .\*.obj
    @rm -rf .\release

# REBUILD
rebuild: clean $(TARGET)
