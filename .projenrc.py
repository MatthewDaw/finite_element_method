from projen.python import PythonProject

project = PythonProject(
    author_email="md@getchief.com",
    author_name="Matthew Daw",
    module_name="finite_element_method",
    name="finite_element_method",
    version="0.1.0",
)

project.synth()