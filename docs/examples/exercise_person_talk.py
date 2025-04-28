class Person:
    # Initialize the the name attribute, so that, when we call the class (Person)
    # we can pass a name that will be stored in the name attribute
    def __init__(self, name):
        self.name = name

    # Create a method that will enable the person to talk
    def talk(self):

        print(f"Hello, my name is {self.name}!")

# Create a variable of the person that actually talks, and give him the ability of the Person class, and pas the full name:
lex = Person("Lex Otto")

print(f"Lex's name is: {lex.name}") # print object lex.name (we attributed the the person class to the name lex)

# Now let Lex introduce himself by calling the method talk in the class Person that is attributed to lex
lex.talk() # In the person class, that is attributed to the lex object, we wrote a method called talk, that prints the name of the person
# So when we call lex.talk() it will print the name of the person that is stored in the name attribute of the class Person
