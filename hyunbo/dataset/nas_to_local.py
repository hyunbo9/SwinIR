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
        nas_to_local(folder)

def nas_to_local(folder):
    
    # if folder is already exist.
    if os.path.exists(folder):
        if input(f"{folder} is already exist. do you want remove?\n") == 'y':
            shutil.rmtree(folder)
        print(f"finish removing {folder}")

    print(f"##### copy progress in {folder} #####")
    shutil.copytree(os.path.join(NAS_ADDRESS, folder), folder)
    print("##### copy progress finish #####")

if __name__ == "__main__":
    main()