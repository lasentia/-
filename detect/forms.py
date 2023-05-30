from django import forms

## 인원 수 입력 받는 클래스 
class PopulationForm(forms.Form):
    count = forms.IntegerField(label="제한할 인원 수를 입력하시오.", min_value=0)