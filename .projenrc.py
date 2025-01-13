"""Projen configuration file."""

from projen.python import PythonProject

AUTHORS = ["Matthew Daw"]
AUTHOR_EMAIL = "md@getChief.com"
AWS_PROFILE_NAME = "sandbox"

ROOT_PROJECT = PythonProject(
    author_email="md@getChief.com",
    author_name="MatthewDaw",
    module_name="",
    name="finite-element-method",
    version="0.1.0",
    poetry=True,
    pytest=False,
    deps=["pre-commit", "python-dotenv@^0.21.0", "python@^3.12.0"],
    dev_deps=[
        "pre-commit",
        "projen",
        "pytest",
        "pytest-mock",
        "pytest-asyncio",
        "pytest-env",
        "pytest-cov",
        "testing.postgresql",
        "moto",
    ],
)

ROOT_PROJECT.add_git_ignore("**/cached_solutions")
ROOT_PROJECT.add_git_ignore("**/.matt_experiment")
ROOT_PROJECT.add_git_ignore("**/.idea")
ROOT_PROJECT.add_git_ignore("**/cdk.out")
ROOT_PROJECT.add_git_ignore("**/.venv*")
ROOT_PROJECT.add_git_ignore("**/.env")
ROOT_PROJECT.add_git_ignore("**/workspace.xml")
ROOT_PROJECT.add_git_ignore("**/misc.xml")
ROOT_PROJECT.add_git_ignore("**/*.xml")
ROOT_PROJECT.add_git_ignore("**/*.gguf")
ROOT_PROJECT.add_git_ignore("**/*.vscode")
ROOT_PROJECT.add_git_ignore("**/local_cache")
ROOT_PROJECT.add_git_ignore(".DS_Store")
ROOT_PROJECT.add_git_ignore("**/.DS_Store")
ROOT_PROJECT.add_git_ignore("yarn.lock")
ROOT_PROJECT.add_git_ignore("**/slack_token.json")

COMMONS_PROJECT_NAME = "common"
COMMONS_PROJECT = PythonProject(
    parent=ROOT_PROJECT,
    author_email=AUTHOR_EMAIL,
    author_name=AUTHORS[0],
    module_name=COMMONS_PROJECT_NAME.replace("-", "_"),
    name=COMMONS_PROJECT_NAME,
    outdir=COMMONS_PROJECT_NAME,
    version="0.0.0",
    description="Common utils.",
    poetry=True,
    deps=[
        "python@^3.12.0",
        "pydantic",
        "matplotlib",
    ],
    dev_deps=[
    ],
)

MESH_GENERATION_PROJECT_NAME = "mesh_generation"
MESH_GENERATION_PROJECT = PythonProject(
    parent=ROOT_PROJECT,
    author_email=AUTHOR_EMAIL,
    author_name=AUTHORS[0],
    module_name=MESH_GENERATION_PROJECT_NAME.replace("-", "_"),
    name=MESH_GENERATION_PROJECT_NAME,
    outdir=MESH_GENERATION_PROJECT_NAME,
    version="0.0.0",
    description="Mesh generation utils.",
    poetry=True,
    deps=[
        "python@^3.12.0",
        "shapely@^2.0.0",
        "pygmsh",
        "numpy",
        "gmsh",
        "adaptmesh",
        "triangle",
        "torch-geometric",
        "gym",
        "torch",
        "pandas",
        "shapely"
    ],
    dev_deps=[
    ],
)


SIMULATIONS_PROJECT_NAME = "simulations"
SIMULATIONS_PROJECT = PythonProject(
    parent=ROOT_PROJECT,
    author_email=AUTHOR_EMAIL,
    author_name=AUTHORS[0],
    module_name=SIMULATIONS_PROJECT_NAME.replace("-", "_"),
    name=SIMULATIONS_PROJECT_NAME,
    outdir=SIMULATIONS_PROJECT_NAME,
    version="0.0.0",
    description="Module for running simulation experiments.",
    poetry=True,
    deps=[
        "python@^3.12.0",
        "shapely",
    ],
    dev_deps=[
    ],
)


DIF_EQ_SETUP_NAME = "dif_eq_setup"
DIF_EQ_SETUP = PythonProject(
    parent=ROOT_PROJECT,
    author_email=AUTHOR_EMAIL,
    author_name=AUTHORS[0],
    module_name=DIF_EQ_SETUP_NAME.replace("-", "_"),
    name=DIF_EQ_SETUP_NAME,
    outdir=DIF_EQ_SETUP_NAME,
    version="0.0.0",
    description="Dif EQ setup.",
    poetry=True,
    deps=[
        "python@^3.12.0",
        "shapely",
    ],
    dev_deps=[
    ],
)


FEM_SOLVER_PROJECT_NAME = "fem_solver"
FEM_SOLVER_PROJECT = PythonProject(
    parent=ROOT_PROJECT,
    author_email=AUTHOR_EMAIL,
    author_name=AUTHORS[0],
    module_name=FEM_SOLVER_PROJECT_NAME.replace("-", "_"),
    name=FEM_SOLVER_PROJECT_NAME,
    outdir=FEM_SOLVER_PROJECT_NAME,
    version="0.0.0",
    description="finite_element_solver.",
    poetry=True,
    deps=[
        "python@^3.12.0",
    ],
    dev_deps=[
    ],
)

ROOT_PROJECT.synth()
COMMONS_PROJECT.synth()
MESH_GENERATION_PROJECT.synth()
SIMULATIONS_PROJECT.synth()
DIF_EQ_SETUP.synth()
FEM_SOLVER_PROJECT.synth()
