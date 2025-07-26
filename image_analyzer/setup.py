from setuptools import setup, find_packages

print("[DEBUG] running setup")

setup(
  name         = 'image_analyzer',
  version      = '0.1'           ,
  packages     = find_packages() ,
  entry_points = {
    'console_scripts': [
      'image-analyzer = image_analyzer.__main__:main'
    ]
  },
  include_package_data = True,
  install_requires     = []  ,
  zip_safe             = False
)

