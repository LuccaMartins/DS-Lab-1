{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv\n",
    "from scipy.io import savemat\n",
    "csv_filename = 'C:/Users/Dell/Documents/ELTE/2nd Semester/Data Science Lab I/DS-Lab-1/Salinas_PCA_8.csv'\n",
    "mat_filename = 'C:/Users/Dell/Documents/ELTE/2nd Semester/Data Science Lab I/DS-Lab-1/Datasets/PCA/Salinas/Salinas_PCA_8.mat'\n",
    "with open(csv_filename, 'r') as csvfile:\n",
    "    csvreader = csv.reader(csvfile)\n",
    "    mat_data = {'data': [row for row in csvreader]}\n",
    "savemat(mat_filename, mat_data)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dslab1",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
