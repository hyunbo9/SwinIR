import shutil
import os 

NAS_ADDRESS = "/media/NAS3/CIPLAB/users/hyunbo/super_resolution/swinir"


def main():
    folder_list = [
        "superresolution",
        "trainsets",
        "testsets",
    ]

    for folder in folder_list:
        move_to_nas(folder)

def move_to_nas(folder):

    # if folder is already exist.
    if os.path.exists(os.path.join(NAS_ADDRESS, folder)):
        if input(f"{folder} is already exist. do you want remove?\n") == 'y':
            shutil.rmtree(os.path.join(NAS_ADDRESS, folder))

    print(f"##### progress in {folder} #####")
    shutil.copytree(folder, os.path.join(NAS_ADDRESS, folder))


if __name__ == "__main__":
    main()