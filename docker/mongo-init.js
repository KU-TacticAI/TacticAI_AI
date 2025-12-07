// MongoDB 초기화 스크립트 - testuser를 admin DB에 저장
db = db.getSiblingDB('admin');

db.createUser({
    user: 'testuser',
    pwd: 'testuserpw',
    roles: [
        { role: 'readWrite', db: 'tacticai' }
    ]
});

print('User testuser created in admin database with access to tacticai');
