import shutil
from glob import glob


folder = "E:\\learning\\Project\\WuYan\\BoxImg\\WuYan多分类标注\\武烟多分类标注\\4"
help_folder = "E:\\learning\\Datasets\\Box\\Test\\help"
# files = glob(folder + "\\*\\*.jpg")
files = glob(help_folder + "\\*jpg")
for index in range(len(files)):
    save_path = f"E:\\learning\\Datasets\\Box\\Test\\Camera1_Label\\ab{index}.jpg"
    shutil.move(files[index], save_path)
    print(f"{save_path}")
# for file in files:
#     save_path = help_folder + "\\" + file.split("\\")[-1]
#     shutil.copy(file, save_path)
