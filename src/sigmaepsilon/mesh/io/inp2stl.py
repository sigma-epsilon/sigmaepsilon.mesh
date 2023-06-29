#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse  # Process the cmd line
import math  # For sqrt
import copy  # For deepcopy

NoneType = type(None)

# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def processCmdLine():
    parser = argparse.ArgumentParser(description="Detect peas.")
    parser.add_argument(
        "--input", help="Name of the input INP file.", nargs=1, type=str, required=True
    )
    parser.add_argument(
        "--output",
        help="Name of the output STL file.",
        nargs=1,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--flip",
        help="Flip normal vectors.",
        nargs="?",
        type=str2bool,
        required=False,
        const=True,
        default=False,
    )
    return parser.parse_args()


def computeLength(aVector):
    return math.sqrt(
        aVector[0] * aVector[0] + aVector[1] * aVector[1] + aVector[2] * aVector[2]
    )


def computeNormal(p1, p2, p3):
    u = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
    v = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]

    normal = [
        (u[1] * v[2]) - (u[2] * v[1]),
        (u[2] * v[0]) - (u[0] * v[2]),
        (u[0] * v[1]) - (u[1] * v[0]),
    ]
    vector_length = computeLength(normal)
    normal[0] /= vector_length
    normal[1] /= vector_length
    normal[2] /= vector_length

    return normal


def isFileExtensionINF(anInputFileName):
    return anInputFileName[len(anInputFileName) - 4 :].lower() == ".inp"


def isFileExtensionSTL(anOutputFileName):
    return anOutputFileName[len(anOutputFileName) - 4 :].lower() == ".stl"


def readInpFile(anInputFileName, aFlipNormalVectorFlag=False):
    # Open the file if it is an INF file
    if not isFileExtensionINF(anInputFileName):
        raise Exception('The file name "' + anInputFileName + '" is not an .inf file')

    input_file = open(anInputFileName, "r")

    # Structure to store the vertices and triangle indices
    vertex_set = []
    triangle_index_set = []
    material_set = []

    # Store the record type, e.g. NODE for vertices, or ELEMENT, TYPE=C3D4 for tetrahedrons
    record_type = None

    vertex_set_offset = 0

    # Read all the lines of the file
    for line in input_file:
        # Change of record type
        if line[0] == "*":
            if line[1] != "*":
                record_type = line.upper()

                # This is a new part
                if record_type[: len("*PART")] == "*PART":
                    vertex_set_offset = len(vertex_set)
                # This is a new mesh
                elif record_type[: len("*ELEMENT, TYPE=C3D4")] == "*ELEMENT, TYPE=C3D4":
                    triangle_index_set.append([])

                    # Material properties are included
                    substring = "ELSET="
                    if substring in record_type:
                        start = record_type.index(substring) + len(substring)
                        end = len(record_type) - 1
                        material_set.append(record_type[start:end].replace("PT_", ""))

        elif record_type[: len("*HEADING")] == "*HEADING":
            pass
        elif (
            record_type[: len("*NODE")] == "*NODE"
            and record_type[: len("*NODE OUTPUT")] != "*NODE OUTPUT"
        ):
            vertex = line.split(",")
            if len(vertex) == 4:
                vertex_set.append(
                    [float(vertex[1]), float(vertex[2]), float(vertex[3])]
                )
            else:
                raise Exception("Cannot interpret this line: ", line)
        # Tetrahedrons, see http://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node32.html
        elif record_type[: len("*ELEMENT, TYPE=C3D4")] == "*ELEMENT, TYPE=C3D4":
            indices = line.split(",")
            if len(indices) == 5:
                triangle_index_set[-1].append(
                    [
                        int(indices[1]) - 1 + vertex_set_offset,
                        int(indices[2]) - 1 + vertex_set_offset,
                        int(indices[3]) - 1 + vertex_set_offset,
                    ]
                )

                triangle_index_set[-1].append(
                    [
                        int(indices[1]) - 1 + vertex_set_offset,
                        int(indices[4]) - 1 + vertex_set_offset,
                        int(indices[2]) - 1 + vertex_set_offset,
                    ]
                )

                triangle_index_set[-1].append(
                    [
                        int(indices[4]) - 1 + vertex_set_offset,
                        int(indices[3]) - 1 + vertex_set_offset,
                        int(indices[2]) - 1 + vertex_set_offset,
                    ]
                )

                triangle_index_set[-1].append(
                    [
                        int(indices[4]) - 1 + vertex_set_offset,
                        int(indices[1]) - 1 + vertex_set_offset,
                        int(indices[3]) - 1 + vertex_set_offset,
                    ]
                )
            else:
                raise Exception("Cannot interpret this line: ", line)
        # Ignore the beam type
        elif record_type[: len("*ELEMENT, TYPE=B31H")] == "*ELEMENT, TYPE=B31H":
            pass
        # Ignore this type
        elif record_type[: len("*ELEMENT, TYPE=S3RS")] == "*ELEMENT, TYPE=S3RS":
            pass
        # Ignore other types
        else:
            pass
            # raise Exception("Does not know how to interpret this line: " + line);

    if aFlipNormalVectorFlag:
        temp = []

        for triangle_set in triangle_index_set:
            temp.append([])

            for triangle in triangle_set:
                p1_index = triangle[0]
                p3_index = triangle[1]
                p2_index = triangle[2]

                temp[-1].append([p1_index, p2_index, p3_index])

        triangle_index_set = temp

    return vertex_set, triangle_index_set, material_set


def writeStlFile(anOutputFileName, aVertexSet, aTriangleIndexSet):
    # Open the file if it is an STL file
    if not isFileExtensionSTL(anOutputFileName):
        raise Exception('The file name "' + anOutputFileName + '" is not an .stl file')

    output_file = open(anOutputFileName, "w")

    i = 0
    output_file.write("solid\n")
    for triangle in aTriangleIndexSet:
        # Save normal here
        p1_index = triangle[0]
        p1 = aVertexSet[p1_index]

        p2_index = triangle[1]
        p2 = aVertexSet[p2_index]

        p3_index = triangle[2]
        p3 = aVertexSet[p3_index]

        normal = computeNormal(p1, p2, p3)

        output_file.write(
            "\tfacet normal "
            + str(normal[0])
            + " "
            + str(normal[1])
            + " "
            + str(normal[2])
            + "\n"
        )
        output_file.write("\touter loop\n")

        output_file.write(
            "\t\tvertex " + str(p1[0]) + " " + str(p1[1]) + " " + str(p1[2]) + "\n"
        )
        output_file.write(
            "\t\tvertex " + str(p2[0]) + " " + str(p2[1]) + " " + str(p2[2]) + "\n"
        )
        output_file.write(
            "\t\tvertex " + str(p3[0]) + " " + str(p3[1]) + " " + str(p3[2]) + "\n"
        )

        output_file.write("\tendloop\n")
        output_file.write("\tendfacet\n")

        i += 1

    output_file.write("endsolid\n")


def getMeshBBox(vertex_set, triangle_index_set):
    min_corner = None
    max_corner = None

    for triangle in triangle_index_set:
        for vertex_id in triangle:
            if isinstance(min_corner, NoneType):
                min_corner = copy.deepcopy(vertex_set[vertex_id])
            else:
                min_corner[0] = min(min_corner[0], vertex_set[vertex_id][0])
                min_corner[1] = min(min_corner[1], vertex_set[vertex_id][1])
                min_corner[2] = min(min_corner[2], vertex_set[vertex_id][2])

            if isinstance(max_corner, NoneType):
                max_corner = copy.deepcopy(vertex_set[vertex_id])
            else:
                max_corner[0] = max(max_corner[0], vertex_set[vertex_id][0])
                max_corner[1] = max(max_corner[1], vertex_set[vertex_id][1])
                max_corner[2] = max(max_corner[2], vertex_set[vertex_id][2])

    return min_corner, max_corner


def getBBox(vertex_set, triangle_index_set):
    # Get the bounding box
    if len(triangle_index_set) == 1:
        min_corner, max_corner = getMeshBBox(vertex_set, triangle_index_set[0])
    else:
        for i, triangle_index in enumerate(triangle_index_set):
            if i == 0:
                min_corner, max_corner = getMeshBBox(vertex_set, triangle_index)
            else:
                temp_min, temp_max = getMeshBBox(vertex_set, triangle_index)

                min_corner[0] = min(min_corner[0], temp_min[0])
                min_corner[1] = min(min_corner[1], temp_min[1])
                min_corner[2] = min(min_corner[2], temp_min[2])

                max_corner[0] = max(max_corner[0], temp_max[0])
                max_corner[1] = max(max_corner[1], temp_max[1])
                max_corner[2] = max(max_corner[2], temp_max[2])

    return min_corner, max_corner


if __name__ == "__main__":
    # Load the arguments from the command line
    args = processCmdLine()

    print("Convert ", args.input[0], " into ", args.output[0])

    if args.flip:
        print("Flip normal vectors")

    # Load the data
    vertex_set, triangle_index_set, material_set = readInpFile(args.input[0], args.flip)
    print("len(vertex_set)", len(vertex_set))
    print("Number of meshes in", args.input[0], ": ", len(triangle_index_set))

    # Get the bounding box
    min_corner, max_corner = getBBox(vertex_set, triangle_index_set)

    # Compute the bounding box
    bbox_range = [
        max_corner[0] - min_corner[0],
        max_corner[1] - min_corner[1],
        max_corner[2] - min_corner[2],
    ]

    print("X Range:", min_corner[0], "to", max_corner[0], "(delta:", bbox_range[0], ")")
    print("Y Range:", min_corner[1], "to", max_corner[1], "(delta:", bbox_range[1], ")")
    print("Z Range:", min_corner[2], "to", max_corner[2], "(delta:", bbox_range[2], ")")

    # There is only one mesh in the INP file
    if len(triangle_index_set) == 1:
        writeStlFile(args.output[0], vertex_set, triangle_index_set[0])
    else:
        for i, triangle_index in enumerate(triangle_index_set):
            output_file_prefix = args.output[0]
            if isFileExtensionSTL(args.output[0]):
                output_file_prefix = output_file_prefix[: len(output_file_prefix) - 4]

            if i < len(material_set):
                output_file_name = (
                    output_file_prefix
                    + "-"
                    + material_set[i]
                    + "-"
                    + str(i + 1)
                    + ".stl"
                )
            else:
                output_file_name = output_file_prefix + "-" + str(i + 1) + ".stl"

            print("Output files:", output_file_name)
            writeStlFile(output_file_name, vertex_set, triangle_index)
