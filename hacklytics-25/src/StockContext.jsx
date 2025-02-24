import {createContext, useState} from 'react';

export const StockContext = createContext();

function StockProvider({children}) {
    const [inputValue, setInputValue] = useState('');

    return (
        <StockContext.Provider value = {{inputValue, setInputValue}}>
            {children}
        </StockContext.Provider>
    )
}

export default StockProvider