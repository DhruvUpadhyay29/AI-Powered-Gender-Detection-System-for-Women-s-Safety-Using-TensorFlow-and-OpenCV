function showRegisterForm() {
    document.getElementById('register-form').style.display = 'block';
    document.getElementById('login-form').style.display = 'none';
}

function showLoginForm() {
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('register-form').style.display = 'none';
}

function register() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    if (email && password) {
        localStorage.setItem(email, password);
        alert('Registered successfully! Please login.');
        document.getElementById('email').value = ''; // Clear input fields
        document.getElementById('password').value = ''; // Clear input fields
    } else {
        alert('Please fill in all fields.');
    }
}

function login() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const storedPassword = localStorage.getItem(email);
    
    if (storedPassword === password) {
        alert('Login successful!');
        window.location.href = 'nextpage.html';
    } else {
        alert('Incorrect email or password.');
    }
}

