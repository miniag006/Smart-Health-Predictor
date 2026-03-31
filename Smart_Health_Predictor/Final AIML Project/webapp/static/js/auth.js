document.addEventListener('DOMContentLoaded',()=>{
  const signup = document.getElementById('signupForm');
  if(signup){
    signup.addEventListener('submit', (e)=>{
      const required = ['full_name','username','password','age','gender','weight','height','email'];
      for(const k of required){
        const el = signup.querySelector(`[name="${k}"]`);
        if(!el || !el.value.trim()){
          e.preventDefault();
          alert('Please fill all fields');
          el && el.focus();
          return;
        }
      }
      const pw = signup.querySelector('[name="password"]').value;
      if(pw.length < 6){ e.preventDefault(); alert('Password must be at least 6 characters'); }
      const age = parseInt(signup.querySelector('[name="age"]').value,10);
      if(isNaN(age) || age<1 || age>120){ e.preventDefault(); alert('Please enter a valid age'); }
    });
  }
});
