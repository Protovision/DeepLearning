#!/bin/bash

th -e "print( torch.load(\"$1\") )" | ./stripcolors.pl

