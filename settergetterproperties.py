#%%
class fruits:
        banana: str
        mango: str
        orange: str
        
class dog(fruits):
        def __init__(self,legs,tail,cute):
                self._legs=legs
                self.tail=tail
                self.cute=cute
        
        def is_dog(self):
                if (self._legs==4) & (self.tail=='yes') & (self.cute=='yes'):
                        return "this is a dog"
                else:
                        return "not dog"
        
        @property
        def legs(self):
                return self._legs
        
        @legs.setter
        def legs(self,value):
                self._legs=value
        
        @property
        def is_fruit(self):
                if (self._legs==4) & (self.tail=='yes') & (self.cute=='yes'):
                        return 3
                else:
                        return 2
        @is_fruit.setter
        def is_fruit(self,value):
                fruits=value


# %%
fido=dog(4,'yes','yes')

# %%
fido.is_dog()

# %%
fido._legs

# %%
fido.tail='no'
# %%
fido.is_dog()
# %%
fido.is_fruit()

# %%
