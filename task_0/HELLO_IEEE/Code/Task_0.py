class MAT_CALC:
    def matsum(self , arr1, arr2):
        rows1 = len(arr1)
        colomns1 = len(arr1[0])
        rows2 = len(arr2)
        colomns2 = len(arr2[0])

        if rows1 != rows2 or colomns1 != colomns2:
            print("can't add matrices !!")

        else:
            add = []
            for i in range(rows1):
                newrow = [0] * colomns1
                add.append(newrow)  # this creates an empty matrix

            for i in range(rows1):
                for j in range(colomns1):
                    add[i][j] = arr1[i][j] + arr2[i][j]

        return add

    def matsub(self,arr1, arr2):
        rows1 = len(arr1)
        colomns1 = len(arr1[0])
        rows2 = len(arr2)
        colomns2 = len(arr2[0])

        if rows1 != rows2 or colomns1 != colomns2:
            print("can't sub matrices !!")

        else:
            sub = [
                [0 for i in range(colomns1)] for i in range(rows1)
            ]  # another way to make empty matrix

            for i in range(rows1):
                for j in range(colomns1):
                    sub[i][j] = arr1[i][j] - arr2[i][j]
        return sub

    def scalarsum(self,scalar, arr):
        rows1 = len(arr)
        colomns1 = len(arr[0])
        scaled = [[0 for i in range(colomns1)] for i in range(rows1)]
        for i in range(rows1):
            for j in range(colomns1):
                scaled[i][j] = arr[i][j] + scalar
        return scaled

    def scalarsub(self,scalar, arr):
        rows1 = len(arr)
        colomns1 = len(arr[0])
        scaled = [[0 for i in range(colomns1)] for i in range(rows1)]
        for i in range(rows1):
            for j in range(colomns1):
                scaled[i][j] = arr[i][j] - scalar
        return scaled

    def matnorm(self,arr):
        rows1 = len(arr)
        colomns1 = len(arr[0])
        max = arr[0][0]
        min = arr[0][0]
        norm = []
        for i in range(rows1):
            newrow = [0] * colomns1
            norm.append(newrow)
            for j in range(colomns1):
                if arr[i][j] > max:
                    max = arr[i][j]
                elif arr[i][j] < min:
                    min = arr[i][j]

        for i in range(rows1):
            for j in range(colomns1):
                norm[i][j] = (arr[i][j] - min) / (max - min)

        return norm

    def matmul(self,arr1, arr2): 
        rows1 = len(arr1)
        colomns1 = len(arr1[0])
        rows2 = len(arr2)
        colomns2 = len(arr2[0])

        if colomns1 != rows2:
            print("can't mult matrices !!")
        else:
            mult = [[0 for i in range(colomns2)] for i in range(rows1)] 

            for i in range(rows1):
                for j in range(colomns2):
                    temp = 0
                    for k in range(colomns1):
                        temp += arr1[i][k] * arr2[k][j] 
                    mult[i][j] = temp 
        return mult

