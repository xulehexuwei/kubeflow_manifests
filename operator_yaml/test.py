class Solution:
    def removeElement(self, nums: list, val: int) -> int:

        n = 0
        for i in range(len(nums)):
            print(nums)
            if nums[i + n] == val:
                for j in range(i, len(nums) - 1 - n):
                    nums[j] = nums[j + 1]
                n += 1

        end = len(nums) - n
        nums = nums[:end]
        print(n)
        print(nums)
        return len(nums)


if __name__ == '__main__':
    Solution().removeElement([0,1,2,2,3,0,4,2], 2)